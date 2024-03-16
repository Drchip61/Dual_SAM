from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry
import seaborn as sns
import math
from utils_tycon import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file
from segment_anything.modeling.mask_decoder import desam,Global_adapter
from icecream import ic
import torchvision.models as tm
from typing import List, Tuple, Type
from typing import Optional, Tuple, Type
import matplotlib.pyplot as plt
class mask_prompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = tm.vgg16_bn(pretrained=True).features[:43]
        self.down = nn.Conv2d(512,256,1,1,0)
    def forward(self,x):
        x = self.vgg(x)
        x = self.down(x)
        return x

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.LayerNorm(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x).permute(0,2,3,1)).permute(0,3,1,2))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class QuickGELU(nn.Module):
    def forward(self,x:torch.Tensor):
        return x*torch.sigmoid(1.702*x)
class adapter(nn.Module):
    def __init__(self,c=768,r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c,c//r,bias=True),QuickGELU(),nn.Linear(c//r,c,bias=True))
        #self.conv = nn.Sequential(Conv(c,c//r,3,1),Conv(c//r,c,3,1))
        self.IN = nn.LayerNorm(c)
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6) 
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)
    
    def forward(self,x):
        ori = x
        b,h,w,c = x.size()
        out1 = self.IN(x.view(b,h*w,c))
        out = self.fc(out1)
        #conv_out1 = x.permute(0,3,1,2)
        #conv_out = self.conv(conv_out1).permute(0,2,3,1)
        return ori+out.view(b,h,w,c)#+conv_out

class _LoRA_qkv(nn.Module):

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()
        self.adapter1 = nn.ModuleList([adapter() for i in range(12)])#adapter()
        self.adapter2 = nn.ModuleList([adapter() for i in range(12)])#adapter()
        #self.mask_prompt = mask_prompt()
        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        self.desam = desam()
        self.global_adapter = Global_adapter()
        #print(self.desam)
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        #print('success')
        # lets freeze first
        #print(sam_model.image_encoder.pos_embed)
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False
            #print(param)
            #break
        


        
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        
        #print(sam_model.image_encoder.keys())
        self.reset_parameters()
        self.sam = sam_model
        

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input,gamma_input, multimask_output, image_size,fake_index):
        image_patch = self.sam.image_encoder.patch_embed(batched_input)
        gamma_patch = self.sam.image_encoder.patch_embed(gamma_input)
        image_patch = image_patch + self.sam.image_encoder.pos_embed
        gamma_patch = gamma_patch + self.sam.image_encoder.pos_embed
        
        dense_prompt_embeddings = batched_input+gamma_input
        prompt1 = self.global_adapter.patch_embed(dense_prompt_embeddings)+self.global_adapter.pos_embed
        #prompt = []
        #prompt.append(prompt1)
        image_embeddings = []
        gamma_embeddings = []
        for i in range(12):
            image_patch = self.sam.image_encoder.blocks[i](image_patch,self.adapter1[i])
            gamma_patch = self.sam.image_encoder.blocks[i](gamma_patch,self.adapter2[i])
            if (i+1)%3 == 0:
                #print((i+1)//3)
                prompt1,prompt2,prompt3 = self.global_adapter.blocks[(i+1)//3-1](prompt1,image_patch,gamma_patch)

                image_embeddings.append((image_patch+self.global_adapter.gated[(i+1)//3-1]*prompt2))
                gamma_embeddings.append((gamma_patch+self.global_adapter.gated1[(i+1)//3-1]*prompt3))

                
        
        de_in = image_embeddings
        de_gamma = gamma_embeddings

        src = self.sam.image_encoder.neck(image_patch.permute(0,3,1,2))
        src1 = self.sam.image_encoder.neck(gamma_patch.permute(0,3,1,2))

        # src_111 = src.flatten(-2)  # b c h w
        #
        # src_111avg = torch.mean(src_111, dim=1)
        # src_222 = src1.flatten(-2)  # b c h w
        # src_222avg = torch.mean(src_222, dim=1)

        image_pe = torch.rand(1, 3)
        res = self.desam(src, src1, de_in, de_gamma, dense_prompt_embeddings, image_pe)

        # src_1after = res[0][3].flatten(-2)#.permute(0, 2, 1)  # b c h w
        #
        # src_1afteravg = torch.mean(src_1after, dim=1)
        # src_2after = res[1][3].flatten(-2)#.permute(0, 2, 1)  # b c h w
        # src_2afteravg = torch.mean(src_2after, dim=1)
        #
        # src_1after1 = res[0][2].flatten(-2)  # .permute(0, 2, 1)  # b c h w

        # src_1afteravg1 = torch.mean(src_1after1, dim=1)
        # src_2after1 = res[1][2].flatten(-2)  # .permute(0, 2, 1)  # b c h w
        # src_2afteravg1 = torch.mean(src_2after1, dim=1)
        #
        # src_1after2 = res[0][1].flatten(-2)  # .permute(0, 2, 1)  # b c h w
        #
        # src_1afteravg2 = torch.mean(src_1after2, dim=1)
        # src_2after2 = res[1][1].flatten(-2)  # .permute(0, 2, 1)  # b c h w
        # src_2afteravg2 = torch.mean(src_2after2, dim=1)
        #
        #
        # self.visualize_similarity(src_1afteravg, src_2after,src_1afteravg1, src_2after1, src_1afteravg2, src_2after2,pattern='r2n')

        #self.visualize_similarity(src_222avg, src_111, src_2afteravg, src_1after, pattern='r2n')

        return res

    def visualize_similarity(self, a_before_cls, b_before_patch, a_after_cls, b_after_patch,c,d,
                             pattern=None):
        a_before_cls = a_before_cls.unsqueeze(1)
        similarities_ori = torch.nn.functional.cosine_similarity(a_before_cls, b_before_patch, dim=-1)
        similarities_ori = torch.mean(similarities_ori, dim=1).squeeze().cpu().detach().numpy()

        a_after_cls = a_after_cls.unsqueeze(1)
        similarities = torch.nn.functional.cosine_similarity(a_after_cls, b_after_patch, dim=-1)
        similarities = torch.mean(similarities, dim=1).squeeze().cpu().detach().numpy()


        c = c.unsqueeze(1)
        similarities_1 = torch.nn.functional.cosine_similarity(c, d, dim=-1)
        similarities_1 = torch.mean(similarities_1, dim=1).squeeze().cpu().detach().numpy()
        # Set Seaborn style
        sns.set(style="whitegrid")

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot KDE curves for "before" and "after" fusion
        sns.kdeplot(similarities, color='r', label='With PMS', ax=ax, multiple="stack")
        sns.kdeplot(similarities_ori, color='g', label='Without PMS', ax=ax, multiple="stack")


        if pattern == 'r2t':
            sign = 'R2T'
        elif pattern == 'r2n':
            sign = 'R2N'
        elif pattern == 'n2t':
            sign = 'N2T'
        elif pattern == 'n2r':
            sign = 'N2R'
        elif pattern == 't2r':
            sign = 'T2R'
        elif pattern == 't2n':
            sign = 'T2N'
        plt.title("Similarity Distribution of Dual Branch", fontsize=18, fontweight='bold')
        plt.xlabel("Cosine Similarity", fontsize=16, fontweight='bold')
        plt.ylabel("Density", fontsize=16, fontweight='bold')

        # Add a legend to distinguish "before" and "after" fusion
        plt.legend(loc='upper left', fontsize=17)

        plt.show()



