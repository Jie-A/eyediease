import torch
from einops import rearrange
from torch import nn

from self_attention_cv.pos_embeddings.relative_pos_enc_qkv import Relative2DPosEncQKV


def _conv1d1x1(in_channels, out_channels):
    """1D convolution with kernel size of 1 followed by batch norm"""
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                         nn.BatchNorm1d(out_channels))


class AxialAttention(nn.Module):
    def __init__(self, dim, in_channels=128, heads=8, dim_head_kq=8):
        """
        Fig.1 page 6 in Axial DeepLab paper

        Args:
            in_channels: the channels of the feature map to be convolved by 1x1 1D conv
            heads: number of heads
            dim_head_kq: inner dim
        """
        super().__init__()  
        self.dim_head = in_channels // heads
        self.dim = dim

        self.heads = heads

        self.dim_head_v = self.dim_head  # d_out
        self.dim_head_kq = dim_head_kq
        self.qkv_channels = self.dim_head_v + self.dim_head_kq * 2
        self.to_qvk = _conv1d1x1(in_channels, self.heads * self.qkv_channels)

        # Position embedding 2D
        self.RelativePosEncQKV = Relative2DPosEncQKV(dim, self.dim_head_v, self.dim_head_kq)

        # Batch normalization - not common, but we dont need to scale down the dot products this way
        self.attention_norm = nn.BatchNorm2d(heads * 3)
        self.out_norm = nn.BatchNorm1d(in_channels * 2)

    def forward(self, x_in):
        assert x_in.dim() == 3, 'Ensure your input is 4D: [b * width, chan, height] or [b * height, chan, width]'

        # Calculate position embedding -> [ batch*width , qkv_channels,  dim ]
        qkv = self.to_qvk(x_in)

        qkv = rearrange(qkv, 'b (q h) d -> b h q d ', d=self.dim, q=self.qkv_channels, h=self.heads)

        # dim_head_kq != dim_head_v so I cannot decompose with einops here I think
        q, k, v = torch.split(qkv, [self.dim_head_kq, self.dim_head_kq, self.dim_head_v], dim=2)

        r_q, r_k, r_v = self.RelativePosEncQKV()

        # Computations are carried as Fig.1 page 6 in Axial DeepLab paper
        qr = torch.einsum('b h i d, i d j -> b h d j ', q, r_q)
        kr = torch.einsum('b h i d, i d j -> b h d j ', k, r_k)

        dots = torch.einsum('b h i d, b h i j -> b h d j', q, k)

        # We normalize the 3 tensors qr, kr, dots together before element-wise addition
        # To do so we concatenate the tensor heads just to normalize them
        # conceptually similar to scaled dot product in MHSA
        # Here n = len(list)
        norm_dots = self.attention_norm(rearrange(list([qr, kr, dots]), 'n b h d j -> b (h n) d j'))

        # Now we can decompose them
        norm_dots = rearrange(norm_dots, 'b (h n) d j -> n b h d j', n=3)

        # And use einsum in the n=3 axis for element-wise sum
        norm_dots = torch.einsum('n b h d j -> b h d j', norm_dots)

        # Last dimension is used softmax and matrix multplication
        attn = torch.softmax(norm_dots, dim=-1)
        # Matrix multiplication will be performed in the dimension of the softmax! Attention :)
        out = torch.einsum('b h d j,  b h i j -> b h i d', attn, v)

        # Last embedding of v
        kv = torch.einsum('b h d j, i d j -> b h i d ', attn, r_v)

        # To perform batch norm as described in paper,
        # we will merge the dimensions that are != self.dim
        # n = 2 = len(list)
        out = self.out_norm(rearrange(list([kv, out]), 'n b h i d ->  b (n h i ) d'))
        # decompose back output and merge heads
        out = rearrange(out, 'b (n h i ) d ->  n b (h i) d ', n=2, h=self.heads)
        # element wise sum in n=2 axis
        return torch.einsum('n b j i -> b j i', out)


def _conv2d1x1(in_channels, out_channels, stride=1):
    """1x1 convolution for contraction and expansion of the channels dimension
    conv is followed by batch norm"""
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(out_channels))


class AxialAttentionBlock(nn.Module):
    def __init__(self, in_channels, dim, heads=8):
        """
        Axial-attention block implementation as described in:
        paper: https://arxiv.org/abs/2003.07853 , Fig. 2 page 7
        blogpost: TBA
        official code: https://github.com/csrhddlam/axial-deeplab
        Args:
            in_channels:
            dim: token's dim
            heads: the number of distict head representations
            axial_att: whether to use axial att or MHSA
            dim_head: for MHSA only
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        d_in = 128  # hardcoded

        # brings the input channels to 128 feature maps
        self.in_conv1x1 = _conv2d1x1(in_channels, d_in)
        self.out_conv1x1 = _conv2d1x1(d_in, in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.dim_head = d_in // self.heads
        self.height_att = AxialAttention(dim=dim, in_channels=d_in, heads=heads)
        self.width_att = AxialAttention(dim=dim, in_channels=d_in, heads=heads)

    def forward(self, x_in):
        assert x_in.dim() == 4, 'Ensure your input is 4D: [batch,channels, height,width]'
        x = self.in_conv1x1(x_in)
        x = self.relu(x)

        # merge batch dim with width
        x = rearrange(x, 'b c h w -> (b w) c h')
        x = self.height_att(x)
        # decompose width + merge batch with height
        x = rearrange(x, '(b w) c h  -> (b h) c w', w=self.dim)

        x = self.width_att(x)
        x = self.relu(x)

        x = rearrange(x, '(b h) c w -> b c h w', h=self.dim)

        return self.relu(self.out_conv1x1(x) + x_in)

if __name__ == '__main__':
    block = AxialAttentionBlock(64, 32)
    block1 = AxialAttention(32, 64)
    a = torch.rand(64, 64, 32)
    print(block1(a).shape)
