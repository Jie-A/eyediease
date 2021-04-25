"""
@author: Duy Le <leanhduy497@gmail.com>
"""
import torch
import torch.nn as nn

from .model_util import init_weights
from .modules import Spatial_Attention, DropBlock2D

__all__ = ['SA_Unet']

class Unet_DropBlock(nn.Module):
    def __init__(self, in_ch, out_ch, block_size, drop_prob, use_attention=False):
        super(Unet_DropBlock, self).__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            DropBlock2D(drop_prob, block_size),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            Spatial_Attention() if use_attention else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            DropBlock2D(drop_prob, block_size),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.unit(input)


class SA_Unet(nn.Module):
    """
        Spatial Attention U-Net (SA-Unet) is a lightweight segmentation network.
        Showing SOTA results on vessel segmentation task on DRIVE, CHASE_DB1 dataset.
        Using Structured Dropout Convolutional Block or DropBlock to deal with over-fitting problem in fundus images dataset.
        It also used Spatial attention to help the net work focus on important features and suppress unnecessary ones to improve the 
        network's representation capability
        More detail at: https://arxiv.org/abs/2004.03696
    """
    def __init__(self, init_filter=16, n_classes=1, block_size=7, drop_prob=0.1):
        super().__init__()
        self.max_pool1 = nn.MaxPool2d((2,2))
        self.max_pool2 = nn.MaxPool2d((2,2))
        self.max_pool3 = nn.MaxPool2d((2,2))
        self.en1 = Unet_DropBlock(3, init_filter*1, block_size, drop_prob=drop_prob)
        self.en2 = Unet_DropBlock(init_filter*1, init_filter*2, block_size, drop_prob=drop_prob)
        self.en3 = Unet_DropBlock(init_filter*2, init_filter*4, block_size, drop_prob=drop_prob)
        self.en4 = Unet_DropBlock(init_filter*4, init_filter*8, block_size, drop_prob=drop_prob, use_attention=True)
        self.up1 = nn.ConvTranspose2d(init_filter*8, init_filter*4, kernel_size=4, stride=2, padding=1)     
        self.up2 = nn.ConvTranspose2d(init_filter*4, init_filter*2, kernel_size=4, stride=2, padding=1)     
        self.up3 = nn.ConvTranspose2d(init_filter*2, init_filter*1, kernel_size=4, stride=2, padding=1)     
        self.dec1 = Unet_DropBlock(init_filter*8, init_filter*4, block_size, drop_prob=drop_prob)
        self.dec2 = Unet_DropBlock(init_filter*4, init_filter*2, block_size, drop_prob=drop_prob)
        self.dec3 = Unet_DropBlock(init_filter*2, init_filter*1, block_size, drop_prob=drop_prob)
        self.out_conv = nn.Conv2d(init_filter*1, n_classes, kernel_size=1)
        init_weights(self)

    def forward(self, input):
        e_1 = self.en1(input)
        p_1 = self.max_pool1(e_1)
        e_2 = self.en2(p_1)
        p_2 = self.max_pool2(e_2)
        e_3 = self.en3(p_2)
        p_3 = self.max_pool3(e_3)
        e_4 = self.en4(p_3)

        d_1 = self.up1(e_4)
        d_1 = torch.cat([d_1, e_3], dim=1)
        d_1 = self.dec1(d_1)
        d_2 = self.up2(e_3)
        d_2 = torch.cat([d_2, e_2], dim=1)
        d_2 = self.dec2(d_2)
        d_3 = self.up3(e_2)
        d_3 = torch.cat([d_3, e_1], dim=1)
        d_3 = self.dec3(d_3)

        output = self.out_conv(d_3)
        return output


if __name__ == '__main__':
    model = SA_Unet().cpu()
    a = torch.randn(2, 3, 1024, 1024, device='cpu')
    output = model(a)
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(total, trainable)
    print(output.shape)