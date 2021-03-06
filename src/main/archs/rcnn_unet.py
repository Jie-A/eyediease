#Credit: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from .modules import *

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    def __init__(self, img_ch=3, output_ch=1, t=2, pretrained=True):
        super(R2U_Net, self).__init__()
        self.pretrained = pretrained
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
      e1 = self.RRCNN1(x)

      e2 = self.Maxpool(e1)
      e2 = self.RRCNN2(e2)

      e3 = self.Maxpool1(e2)
      e3 = self.RRCNN3(e3)

      e4 = self.Maxpool2(e3)
      e4 = self.RRCNN4(e4)

      e5 = self.Maxpool3(e4)
      e5 = self.RRCNN5(e5)

      d5 = self.Up5(e5)
      d5 = torch.cat((e4, d5), dim=1)
      d5 = self.Up_RRCNN5(d5)

      d4 = self.Up4(d5)
      d4 = torch.cat((e3, d4), dim=1)
      d4 = self.Up_RRCNN4(d4)

      d3 = self.Up3(d4)
      d3 = torch.cat((e2, d3), dim=1)
      d3 = self.Up_RRCNN3(d3)

      d2 = self.Up2(d3)
      d2 = torch.cat((e1, d2), dim=1)
      d2 = self.Up_RRCNN2(d2)

      out = self.Conv(d2)

      return out
