# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import copy
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import List, Tuple, Type
from typing import Optional, Tuple, Type
from .common import LayerNorm2d, MLPBlock
from .cross_attention import Multi_CrossAttention

class NLBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(NLBlock, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        # 相当于计算value
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 相当于计算query
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # 相当于计算key
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # 计算attention map
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        # output
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # 残差连接
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class senet(nn.Module):
    def __init__(self,c=256,r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c,c//r,1,1,0,bias=True),nn.ReLU(),nn.Conv2d(c//r,c,1,1,0,bias=True))
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)

    def forward(self,x):
        res = x
        b,c,h,w=x.size()
        #x = x.view(b,c,h*w)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out+max_out
        x = x*self.sigmoid(out)
        #x = x.view(b,c,h,w)
        return x+res

def autopad(k, d,p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    if d != 1:
        p = d
    return p
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, d=1,p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, d,p), dilation=d,groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DeConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, 2, 2, 0)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=7, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s),g=c_)
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=c2)
        #self.add = shortcut and c1 == c2

    def forward(self, x):
        return x+self.cv2(self.cv1(x))# if self.add else self.cv2(self.cv1(x))

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int=768,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        #print(x.size())
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
class senet1(nn.Module):
    def __init__(self,c=256,r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c,c//r,1,1,0,bias=True),nn.ReLU(),nn.Conv2d(c//r,c,1,1,0,bias=True))
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()
        self.pw1 = Conv(c*2,c,1,1)
        self.dw = CrossConv(c1=c,c2=c)
        self.pw2 = Conv(c,c,3,1,2)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)

    def forward(self,x):
        x = self.pw1(x)
        #x = self.dw(x)
        #x = self.pw2(x)
        res = x
        b,c,h,w=x.size()
        #x = x.view(b,c,h*w)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = max_out+avg_out
        x = x*self.sigmoid(out)
        x = x.view(b,c,h,w)
        return x+res
class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int=768,
        num_heads: int=12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.CA = Multi_CrossAttention(768,768)
        
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        #self.senet = senet(768)
        self.mlp2 = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size
        self.pw1 = Conv(768,768,1,1)
        self.pw2 = Conv(768,768,1,1)

    def forward(self, x,y,z) -> torch.Tensor:
        
        shortcut1 = x
        #print(type(x))
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut1 + x
        x = x + self.mlp2(self.norm2(x))
        
        #prompt1 = self.senet(x.permute(0,3,1,2)).permute(0,2,3,1)
        prompt1 = x
        shortcut = x
        x = self.norm1(x)#.permute(0,3,1,2)
        y = self.norm1(y).permute(0,3,1,2)
        z = self.norm1(z).permute(0,3,1,2)
        # Window partition
        
        x = self.CA(y,z,x).permute(0,2,3,1)
        

        x = shortcut + x
        prompt2_ = x + self.mlp(self.norm2(x))
        prompt2_ = prompt2_.permute(0,3,1,2)
        prompt2 = self.pw1(prompt2_).permute(0,2,3,1)
        prompt3 = self.pw2(prompt2_).permute(0,2,3,1)
        #x = x.permute(0,3,1,2)
        #x = self.senet(x).permute(0,2,3,1)
       
        return prompt1,prompt2,prompt3
#a = torch.rand(1,32,32,768)
#model = Block()
#print(model(a).size())


class conv_block(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()

        #self.fuse = NLBlock(out_c,out_c//16)
        self.se = senet(c = out_c)
        self.pw1 = Conv(in_c,out_c,1,1)
        #self.dw = CrossConv(c1=out_c,c2=out_c)
        self.pw2 = Conv(out_c,out_c,3,1,2)
    def forward(self,x):
        x = self.pw1(x)
        #res = x
        x1 = self.se(x)
        x1 = self.pw2(x1)
        #x2 = self.dw(x)
        #fuse = x1+x2
        #fuse = torch.cat([x1,x2],dim=1)
        #res_fuse = self.pw2(fuse)+res
        return x1#res_fuse

class conv_up(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.conv_up = nn.ConvTranspose2d(in_c,out_c,2,2,0)
        self.conv_fu = Conv(out_c,out_c)
    def forward(self,x):
        x = self.conv_up(x)
        x = self.conv_fu(x)
        return x

class conv_up0(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.conv_up = nn.ConvTranspose2d(in_c,out_c,1,1,0)
        self.conv_fu = Conv(out_c,out_c)
    def forward(self,x):
        x = self.conv_up(x)
        x = self.conv_fu(x)
        return x

class conv_pre(nn.Module):
    def __init__(self,in_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.pre = nn.Conv2d(in_c,8,1,1,0)
        #self.conv = CrossConv(in_c,out_c)
    def forward(self,x):
        #x = self.conv(x)
        x = self.pre(x)
        return x

class conv_up_pre(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.up = Conv(in_c,out_c)
        self.pre = nn.Conv2d(out_c,8,1,1,0)
        #self.conv = CrossConv(in_c,out_c)
    def forward(self,x):
        x = self.up(x)
        x = self.pre(x)
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
      
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class Global_adapter(nn.Module):
    def __init__(self,img_size=512,patch_size=16,embed_dim=768,in_chans=3):
        super().__init__()
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
        )
        '''
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
        )
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.blocks = nn.ModuleList()
        for i in range(4):
            block = Block()
            self.blocks.append(block)
        '''
        self.gated = []
        self.gated.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda())
        self.gated.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda())
        self.gated.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda())
        self.gated.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda())

        self.gated1 = []
        self.gated1.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda())
        self.gated1.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda())
        self.gated1.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda())
        self.gated1.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda())
    def forward(self,x,ori,gamma,fake_index):

        return ori,gamma


class Global_adapter1(nn.Module):
    def __init__(self, img_size=512, patch_size=16, embed_dim=768, in_chans=3):
        super().__init__()



    def forward(self, x, ori, gamma, fake_index):

        return ori, gamma


class desam(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.Global_adapter = Global_adapter1()


        self.de12 = conv_up0(768,256) 
        self.de9 = conv_up(768,128)
        self.de6 = nn.Sequential(conv_up(768,128),conv_up(128,64))
        self.de3 = nn.Sequential(conv_up(768,128),conv_up(128,64),conv_up(64,32))

        
        self.de121 = conv_up0(768,256) 
        self.de91 = conv_up(768,128)
        self.de61 = nn.Sequential(conv_up(768,128),conv_up(128,64))
        self.de31 = nn.Sequential(conv_up(768,128),conv_up(128,64),conv_up(64,32))


        self.up1 = conv_up(256,256)
        self.up2 = conv_up(128,128)
        self.up3 = conv_up(64,64)

        self.up11 = conv_up(256,256)
        self.up22 = conv_up(128,128)
        self.up33 = conv_up(64,64)

        self.fu1 = conv_block(512,256)
        self.fu2 = conv_block(384,128)
        self.fu3 = conv_block(192,64)
        self.fu4 = conv_block(96,32)
 
        self.fu11 = conv_block(512,256)
        self.fu22 = conv_block(384,128)
        self.fu33 = conv_block(192,64)
        self.fu44 = conv_block(96,32)
     
        self.pre4 = nn.Sequential(nn.ConvTranspose2d(32,32,2,2,0),conv_pre(32))
        self.pre3 = nn.Sequential(nn.ConvTranspose2d(64,32,2,2,0),conv_up_pre(32,32)) 
        self.pre2 = nn.Sequential(nn.ConvTranspose2d(128,32,2,2,0),conv_up_pre(32,32)) 
        self.pre1 = nn.Sequential(nn.ConvTranspose2d(256,32,2,2,0),conv_up_pre(32,32))  
        
        self.pre44 = nn.Sequential(nn.ConvTranspose2d(32,32,2,2,0),conv_pre(32))
        self.pre33 = nn.Sequential(nn.ConvTranspose2d(64,32,2,2,0),conv_up_pre(32,32)) 
        self.pre22 = nn.Sequential(nn.ConvTranspose2d(128,32,2,2,0),conv_up_pre(32,32)) 
        self.pre11 = nn.Sequential(nn.ConvTranspose2d(256,32,2,2,0),conv_up_pre(32,32)) 

    def forward(self,mask_in_ori,mask_in_gamma,mask_embed,gamma_embed,dense_prompt_embeddings,fake_index):
 
        ori_prompt,gamma_prompt = self.Global_adapter(dense_prompt_embeddings,gamma_embed,mask_embed,fake_index)

        ori_de12 = self.de12(ori_prompt[-1].permute(0,3,1,2))#256,32,32
        ori_de9 = self.de9(ori_prompt[-2].permute(0,3,1,2))#128,64,64
        ori_de6 = self.de6(ori_prompt[-3].permute(0,3,1,2))#64,128,128
        ori_de3 = self.de3(ori_prompt[-4].permute(0,3,1,2))#32,256,256

        gamma_de12 = self.de121(gamma_prompt[-1].permute(0,3,1,2))#256,32,32
        gamma_de9 = self.de91(gamma_prompt[-2].permute(0,3,1,2))#128,64,64
        gamma_de6 = self.de61(gamma_prompt[-3].permute(0,3,1,2))#64,128,128
        gamma_de3 = self.de31(gamma_prompt[-4].permute(0,3,1,2))#32,256,256

        
        ori_mask1 = torch.cat([mask_in_ori,ori_de12],dim=1)
        ori_mask1_ = self.fu1(ori_mask1)#256,32,32
        ori_mask1 = self.up1(ori_mask1_)
        ori_mask2 = torch.cat([ori_mask1,ori_de9],dim=1)
        ori_mask2_ = self.fu2(ori_mask2)#128,64,64
        ori_mask2 = self.up2(ori_mask2_)
        ori_mask3 = torch.cat([ori_mask2,ori_de6],dim=1)
        ori_mask3_ = self.fu3(ori_mask3)#64,128,128
        ori_mask3 = self.up3(ori_mask3_)
        ori_mask4 = torch.cat([ori_mask3,ori_de3],dim=1)
        ori_mask4 = self.fu4(ori_mask4)#32,256,256
 
        gamma_mask1 = torch.cat([mask_in_gamma,gamma_de12],dim=1)
        gamma_mask1_ = self.fu11(gamma_mask1)#256,32,32
        gamma_mask1 = self.up11(gamma_mask1_)
        gamma_mask2 = torch.cat([gamma_mask1,gamma_de9],dim=1)
        gamma_mask2_ = self.fu22(gamma_mask2)#128,64,64
        gamma_mask2 = self.up22(gamma_mask2_)
        gamma_mask3 = torch.cat([gamma_mask2,gamma_de6],dim=1)
        gamma_mask3_ = self.fu33(gamma_mask3)#64,128,128
        gamma_mask3 = self.up33(gamma_mask3_)
        gamma_mask4 = torch.cat([gamma_mask3,gamma_de3],dim=1)
        gamma_mask4 = self.fu44(gamma_mask4)#32,256,256
   
        ori_mask_pre4 = self.pre4(ori_mask4)
        ori_mask_pre3 = self.pre3(ori_mask3_)
        ori_mask_pre2 = self.pre2(ori_mask2_)
        ori_mask_pre1 = self.pre1(ori_mask1_)

        ori_pre = [ori_mask_pre4,ori_mask_pre3,ori_mask_pre2,ori_mask_pre1]

        gamma_mask_pre4 = self.pre44(gamma_mask4)
        gamma_mask_pre3 = self.pre33(gamma_mask3_)
        gamma_mask_pre2 = self.pre22(gamma_mask2_)
        gamma_mask_pre1 = self.pre11(gamma_mask1_)

        gamma_pre = [gamma_mask_pre4,gamma_mask_pre3,gamma_mask_pre2,gamma_mask_pre1]

        return ori_pre,gamma_pre

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:

        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 2#num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        
        self.desam = desam()
        
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, 4, iou_head_depth
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        gamma_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            gamma_embeddings=gamma_embeddings,
            image_pe=multimask_output,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        gamma_embeddings: torch.Tensor,
        image_pe: float,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        de_in = image_embeddings[1:]
        de_gamma = gamma_embeddings[1:]
        image_embeddings = image_embeddings[0]
        gamma_embeddings = gamma_embeddings[0]


        #output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        #output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        #tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        #src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        #print(src.size())
        #print(dense_prompt_embeddings.size())
        #src = src + dense_prompt_embeddings
        #pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        #b, c, h, w = src.shape

        #src1 = torch.repeat_interleave(gamma_embeddings, tokens.shape[0], dim=0)
        #print(src.size())
        #print(dense_prompt_embeddings.size())
        #src1 = src1 + dense_prompt_embeddings
        #pos_src1 = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = image_embeddings.shape
        #print(b,c,h,w)

        # Run the transformer
        #hs, src = self.transformer(src, pos_src, tokens)
        #hs1, src1 = self.transformer(src1, pos_src, tokens)
        
        #iou_token_out = hs[:, 0, :]
        #mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = image_embeddings#.transpose(1, 2).view(b, c, h, w)
        src1 = gamma_embeddings#.transpose(1, 2).view(b, c, h, w)
        #src_ = src+src1
        res = self.desam(src,src1,de_in,de_gamma,dense_prompt_embeddings,image_pe)
        #upscaled_embedding = self.output_upscaling(src)
        #hyper_in_list: List[torch.Tensor] = []
        #for i in range(self.num_mask_tokens):
        #    hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        #hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        #b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        #masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Generate mask quality predictions
        iou_pred = 0#self.iou_prediction_head(iou_token_out)

        return res, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
