import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor

from util import set_trainable, init_weights

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class conv_block(nn.Module):
    """
    Convolution Block 
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, use_bilinear=True):
        super(up_conv, self).__init__()
        if use_bilinear:
            module_list = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
        else:
            module_list = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))

        self.up = module_list

    def forward(self, x):
        x = self.up(x)
        return x


class AttU_Net(nn.Module):
    """
    Attention Unet implementation from scratch
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(
            F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(
            F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(
            F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(
            filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        init_weights(self)

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class Attention_Unet(nn.Module):
    """
        Attention Unet with pretrained model.
        Resnet18, resnet34, resnet50, resnet101, wide_resnet
    """
    def __init__(self, in_ch, out_ch, backbone='resnet50', pretrained=True, freeze_bn=False, freeze_backbone=False):
        super(Attention_Unet, self).__init__()
        model = torch.hub.load(
            'pytorch/vision', backbone, pretrained=pretrained)

        self.initial = list(model.children())[:4]
        if in_ch != 3:
            self.initial[0] = nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        filters = [256, 512, 1024, 2048]

        self.conv5 = conv_block(filters[3], 192)
        self.up_conv5 = up_conv(192, 128)
        self.att4 = Attention_block(
            F_g=128, F_l=filters[2], F_int=filters[1])

        self.conv4 = conv_block(filters[2] + 128, 128)
        self.up_conv4 = up_conv(128, 96)
        self.att3 = Attention_block(
            F_g=96, F_l=filters[1], F_int=filters[0])

        self.conv3 = conv_block(filters[1] + 96, 96)
        self.up_conv3 = up_conv(96, 64)
        self.att2 = Attention_block(
            F_g=64, F_l=filters[0], F_int=filters[0])

        self.conv2 = conv_block(filters[0] + 64, 64)
        self.up_conv2 = up_conv(64, 48)

        self.conv1 = conv_block(48, 48)
        self.up_conv1 = up_conv(48, 32)

        self.conv0 = conv_block(32, 32)
        self.final = conv_block(32, out_ch)

        init_weights(self)
        
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #Encode
        #x shape B,C,W,H, C=3
        #init shape B, C1, W//4, H//4, C1=64
        init = self.initial(x)
        #e1 shape B, C2, W//4, H//4, C2= 256
        e1 = self.layer1(init)
        #e2 shape B, C3, W//8, H//8, C3=512
        e2 = self.layer2(e1)
        #e3 shape B, C4, W//16, H//16, C4=1024
        e3 = self.layer3(e2)
        #e4 shape B, C5, W//32, H//32, C5=2048
        e4 = self.layer4(e3)
        
        #Decode
        d4 = self.up_conv5(self.conv5(e4))
        d4 = F.interpolate(d4, size=(e3.size(2), e3.size(3)), mode="bilinear", align_corners=True)
        a4 = self.att4(g=d4, x=e3)
        d3 = torch.cat([a4, d4], dim=1)

        d3 = self.up_conv4(self.conv4(d3))
        d3 = F.interpolate(d3, size=(e2.size(2), e2.size(3)), mode="bilinear", align_corners=True)
        a3 = self.att3(g=d3, x=e2)
        d2 = torch.cat([a3, d3], dim=1)

        d2 = self.up_conv3(self.conv3(d2))
        d2 = F.interpolate(d2, size=(e1.size(2), e1.size(3)), mode="bilinear", align_corners=True)
        a2 = self.att2(g=d2, x=e1)
        d1 = torch.cat([a2, d2], dim=1)


        d0 = self.up_conv2(self.conv2(d1))

        final = self.up_conv1(self.conv1(d0))

        # if the input is not divisible by the output stride
        if final.size(2) != H or final.size(3) != W:
            final = F.interpolate(final, size=(H, W), mode="bilinear", align_corners=True)

        final = self.final(self.conv0(final))

        return final

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
