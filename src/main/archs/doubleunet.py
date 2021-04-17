#Convert from TF source code from https://github.com/DebeshJha/2020-CBMS-DoubleU-Net/blob/master/model.py
"""
@author: Duy Le <leanhduy497@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from modules import *   
from model_util import init_weights

class Encoder1(nn.Module):
    def __init__(self, freeze_bn=True, backbone='resnext50d_32x4d', freeze_backbone=False, pretrained=True):
        super(Encoder1, self).__init__()
        if backbone:
            self.encoder = timm.create_model(backbone, features_only=True, pretrained=pretrained)
            self.filters = self.encoder.feature_info.channels()
            if freeze_bn:
                self.freeze_bn()
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, input):
        encoder_features = self.encoder(input)
        return encoder_features


class Decoder1(nn.Module):
    def __init__(self, n_classes, encoder_channels):
        super(Decoder1, self).__init__()
        self.encoder_channels = encoder_channels[::-1]
        self.decoder_output = nn.ModuleList()
        array_1 = self.encoder_channels[:-1]
        array_2 = self.encoder_channels[1:]

        for i, (in_ch, out_ch) in enumerate(zip(array_1, array_2)):
            next_up = nn.Sequential(
                Up(in_ch, out_ch),
                SE_Block(out_ch, r=8)
            )
            self.decoder_output.append(next_up)
        self.clf = OutConv(self.encoder_channels[-1], n_classes)
        init_weights(self)

    def forward(self, encoder_features):      
        reverse_features = encoder_features[::-1]  
        up_decode = reverse_features[0]
        for i, feature in enumerate(reverse_features[1: ]):
            out_decode = self.decoder_output[i](up_decode, feature)
            up_decode = out_decode
        final = self.clf(up_decode)
        return final


class Encoder2(nn.Module):
    def __init__(self, in_ch, encoder1_channels):
        super(Encoder2, self).__init__()
        self.encoder1_channels = encoder1_channels  
        self.blocks = nn.ModuleList()
        array_1 = self.encoder1_channels[:-1]
        array_2 = self.encoder1_channels[1:]
        self.blocks.append(Down(in_ch, array_1[0]))
        for i, (f_in, f_out) in enumerate(zip(array_1, array_2)):
            self.blocks.append(Down(f_in, f_out))
        init_weights(self)

    def forward(self, inputs):
        encoder_features = self.blocks(inputs)
        return encoder_features

class Up_1(Up):
    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)

class Decoder2(nn.Module):
    def __init__(self, n_classes, dropout, encoder1_channels, encoder2_channels):
        super(Decoder2, self).__init__()
        self.encoder1_channels = encoder1_channels[::-1]
        self.encoder2_channels = encoder2_channels[::-1]
        self.decoder_output = nn.ModuleList()
        array_1 = self.encoder1_channels[:-1]
        array_2 = self.encoder1_channels[1:]

        for i, (in_ch, out_ch) in enumerate(zip(array_1, array_2)):
            next_up = nn.Sequential(
                Up_1(in_ch, out_ch),
                SE_Block(out_ch, r=8)
            )
            self.decoder_output.append(next_up)
        self.clf = OutConv(self.encoder2_channels[-1], n_classes)
        self.dropout = nn.Dropout2d(dropout)
        init_weights(self)

    def forward(self, encoder1_features, encoder2_features):      
        reverse_features_1 = encoder1_features[::-1]  
        reverse_features_2 = encoder2_features[::-1]
        up_decode = reverse_features_2[0]
        for i, (feature1, feature2) in enumerate(zip(reverse_features_1[1: ], reverse_features_2[1: ])):
            out_decode = self.decoder_output[i](up_decode, feature1, feature2)
            up_decode = out_decode
        final = self.dropout(up_decode)
        final = self.clf(final)
        return final


class Double_Unet(nn.Module):
    def __init__(self, in_ch, n_classes, dropout, encoder1: nn.Module):
        super(Double_Unet, self).__init__()
        self.encoder1 = encoder1
        encoder1_channels = self.encoder1.filters
        self.decoder1 = Decoder1(n_classes, encoder1_channels)
        self.encoder2 = Encoder2(in_ch, encoder1_channels)
        self.decoder2 = Decoder2(n_classes, dropout, encoder1_channels, encoder1_channels)
        self.aspp1 = ASPP(encoder1_channels[-1], 16, encoder1_channels[-1])
        self.aspp2 = ASPP(encoder1_channels[-1], 16, encoder1_channels[-1])

    def forward(self, input):
        encoder1_f = self.encoder1(input)
        x = self.aspp1(encoder1_f[-1])
        encoder1_f[-1] = x
        output1 = self.decoder1(encoder1_f)

        se_inputs = input*output1

        encoder2_f = self.encoder2(se_inputs)
        x = self.aspp2(encoder2_f[-1])
        encoder2_f[-1] = x
        output2 = self.decoder2(encoder1_f, encoder2_f)
        sum_output = torch.sum([output1, output2], dim=1)
        
        return sum_output

if __name__ == '__main__':
    device = 'cpu'
    encoder = Encoder1(backbone='tf_efficientnet_b2')
    model = Double_Unet(3, 1, 0.2, encoder)
    model = model.to(device)

    a = torch.randn(1, 3, 256, 256, device = device)

    print(model)
    print(model(a).shape)