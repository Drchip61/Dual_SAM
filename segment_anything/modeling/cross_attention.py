from math import sqrt
import torch
import torch.nn as nn


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # use mask
        #attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention, V)
        return attention

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
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class conv_block(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.se = senet(c = out_c)
        self.pw1 = Conv(in_c,out_c,1,1)
        #self.dw = CrossConv(c1=out_c,c2=out_c)
        self.pw2 = Conv(out_c,out_c,1,1)
    def forward(self,x):
        x = self.pw1(x)
        x = self.se(x)
        x = self.pw2(x)
        #x = self.se(x)
        return x

class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query和key，第二个参数用于计算value
    """

    def __init__(self, hidden_size, all_head_size, head_num=8):
        super().__init__()
        self.hidden_size = hidden_size  # 输入维度
        self.all_head_size = all_head_size  # 输出维度
        self.num_heads = head_num  # 注意头的数量
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(all_head_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, all_head_size)

        # normalization
        self.norm = sqrt(all_head_size)
        #self.reduce = conv_block(768*2,768)
    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)

    def forward(self, x, y,z):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q和k的输入，y作为v的输入
        """
        
        batch_size = x.size(0)
        b,c,h,w = x.size() 
        b,C,h,w = y.size()
        #fuse = x+y
        #print(x.size())
        #print(y.size())
        #fuse = fuse.view(b,c,h*w).transpose(-1,-2)
        x = x.view(b,C,h*w).transpose(-1,-2)
        y = y.view(b,C,h*w).transpose(-1,-2)
        z = z.view(b,C,h*w).transpose(-1,-2)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(z).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        #attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s, k_s, v_s)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)
        output = output.transpose(-1,-2)
        output = output.view(b,c,h,w)

        return output
