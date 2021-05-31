"""
@author: Duy Le <leanhduy497@gmail.com>
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules import encoders as E

import timm
from .modules import *
from .model_util import init_weights

__all__ = [
    'Attention_Unet', 
    'Unet_Encoder',
    'resnet50_attunet', 
    'efficientnetb2_attunet', 
    'mobilenetv3_attunet'
]

class Unet_Encoder(nn.Module):
    def __init__(self, freeze_bn=True, backbone='resnext50d_32x4d', freeze_backbone=False, pretrained=True):
        super(Unet_Encoder, self).__init__()
        if backbone:
            self.encoder = timm.create_model(backbone, features_only=True, pretrained=pretrained)
            self.filters = self.encoder.feature_info.channels()
            if freeze_bn:
                self.freeze_bn()
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            init_weights(self)
            pass

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, input):
        encoder_features = self.encoder(input)
        return encoder_features

class Up_Atten(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear = True):
        super(Up_Atten, self).__init__()
        self.atten = Attention_block(F_g=in_ch // 2, F_l=out_ch, F_int=in_ch)
        self.up_conv = DoubleConv(in_ch // 2 + out_ch, out_ch)
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=4, stride=2, padding=1)

    def forward(self, input1, input2):
        d2 = self.up(input1) 
        d1 = self.atten(d2, input2)
        d2 = F.interpolate(d2, size=(d1.size(2), d1.size(3)), mode="bilinear", align_corners=True)
        d = torch.cat([d1, d2], dim=1)
        return self.up_conv(d)

class Unet_Decoder(nn.Module):
    def __init__(self, encoder_channels, n_classes, dropout):
        super(Unet_Decoder, self).__init__()
        self.encoder_channels = encoder_channels[::-1]
        self.n_classes = n_classes  

        self.decoder_output = nn.ModuleList()
        array_1 = self.encoder_channels[:-1]
        array_2 = self.encoder_channels[1:]

        for i, (in_ch, out_ch) in enumerate(zip(array_1, array_2)):
            next_up = Up_Atten(in_ch, out_ch)
            self.decoder_output.append(next_up)
        self.dropout = nn.Dropout2d(dropout)
        self.out_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            OutConv(self.encoder_channels[-1], n_classes)
        )
        init_weights(self)

    def forward(self, encoder_features):      
        reverse_features = encoder_features[::-1]  
        up_decode = reverse_features[0]
        for i, feature in enumerate(reverse_features[1: ]):
            out_decode = self.decoder_output[i](up_decode, feature)
            up_decode = out_decode
        final = self.dropout(up_decode)
        return self.out_conv(final)

class Attention_Unet(nn.Module):
    """
        Attention Unet with pretrained model.
        Resnet18, resnet34, resnet50, resnet101, wide_resnet, ... from timm package
    """
    def __init__(self, n_classes, dropout, encoder: nn.Module):
        super(Attention_Unet, self).__init__()
        
        self.encoder = encoder
        encoder_channels = self.encoder.filters
        self.decoder = Unet_Decoder(encoder_channels, n_classes, dropout)

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)
        H, W = x.size(2), x.size(3)

        #Encode
        encoder_outputs = self.encoder(x)
        #Decode
        final = self.decoder(encoder_outputs)
        # if the input is not divisible by the output stride
        if final.size(2) != H or final.size(3) != W:
            final = F.interpolate(final, size=(H, W), mode="bilinear", align_corners=True)
        
        final = unpad_image_tensor(final, pad)
        return final

def resnet50_attunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False):
    encoder = Unet_Encoder(freeze_bn=freeze_bn, backbone='resnet50', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder)

def efficientnetb2_attunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False):
    encoder = Unet_Encoder(freeze_bn=freeze_bn, backbone='tf_efficientnet_b2', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder)

def mobilenetv3_attunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False):
    encoder = Unet_Encoder(freeze_bn=freeze_bn, backbone='mobilenetv3_large_100', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder)

# class Unet_EncoderResNet(nn.Module):
#     """
#         Pretrained Encoder module for Unet 
#     """
#     def __init__(self, in_ch, backbone='resnet50', pretrained=True, freeze_bn=False, freeze_backbone=False):
#         super(Unet_EncoderResNet, self).__init__()
#         model = torch.hub.load(
#             'pytorch/vision', backbone, pretrained=pretrained)

#         initial = list(model.children())[:4]
#         if in_ch != 3:
#             initial[0] = nn.Conv2d(
#                 in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.initial = nn.Sequential(*initial)

#         # encoder
#         self.layer1 = model.layer1
#         self.layer2 = model.layer2
#         self.layer3 = model.layer3
#         self.layer4 = model.layer4

#         if not pretrained:
#             init_weights(self)

#         if freeze_bn:
#             self.freeze_bn()
#         if freeze_backbone:
#             set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)        

#     def get_filters(self):
#         encoder_filters = [256, 512, 1024, 2048]
#         return encoder_filters

#     def freeze_bn(self):
#         for module in self.modules():
#             if isinstance(module, nn.BatchNorm2d):
#                 module.eval()

#     def forward(self, input):
#         #x shape B,C,W,H, C=3
#         #init shape B, C1, W//4, H//4, C1=64
#         init = self.initial(input)
#         #e1 shape B, C2, W//4, H//4, C2= 256
#         e1 = self.layer1(init)
#         #e2 shape B, C3, W//8, H//8, C3=512
#         e2 = self.layer2(e1)
#         #e3 shape B, C4, W//16, H//16, C4=1024
#         e3 = self.layer3(e2)
#         #e4 shape B, C5, W//32, H//32, C5=2048
#         e4 = self.layer4(e3)
    
#         return [e1, e2, e3, e4]


# class Unet_DecoderResNet(nn.Module):
#     """
#         Decoder module for Unet
#     """
#     def __init__(self, encoder_filters):
#         super(Unet_DecoderResNet, self).__init__()
        
#         self.conv5 = conv_block(encoder_filters[3], 192)
#         self.up_conv5 = up_conv(192, 128)
#         self.att4 = Attention_block(
#             F_g=128, F_l=encoder_filters[2], F_int=encoder_filters[1])

#         self.conv4 = conv_block(encoder_filters[2] + 128, 128)
#         self.up_conv4 = up_conv(128, 96)
#         self.att3 = Attention_block(
#             F_g=96, F_l=encoder_filters[1], F_int=encoder_filters[0])

#         self.conv3 = conv_block(encoder_filters[1] + 96, 96)
#         self.up_conv3 = up_conv(96, 64)
#         self.att2 = Attention_block(
#             F_g=64, F_l=encoder_filters[0], F_int=encoder_filters[0])

#         self.conv2 = conv_block(encoder_filters[0] + 64, 64)
#         self.up_conv2 = up_conv(64, 48)

#         self.conv1 = conv_block(48, 48)
#         self.up_conv1 = up_conv(48, 32)
        
#         init_weights(self)

#     def forward(self, encoder_output):
#         e1, e2, e3, e4 = encoder_output
#         d4 = self.up_conv5(self.conv5(e4))
#         d4 = F.interpolate(d4, size=(e3.size(2), e3.size(3)), mode="bilinear", align_corners=True)
#         a4 = self.att4(g=d4, x=e3)
#         d3 = torch.cat([a4, d4], dim=1)

#         d3 = self.up_conv4(self.conv4(d3))
#         d3 = F.interpolate(d3, size=(e2.size(2), e2.size(3)), mode="bilinear", align_corners=True)
#         a3 = self.att3(g=d3, x=e2)
#         d2 = torch.cat([a3, d3], dim=1)

#         d2 = self.up_conv3(self.conv3(d2))
#         d2 = F.interpolate(d2, size=(e1.size(2), e1.size(3)), mode="bilinear", align_corners=True)
#         a2 = self.att2(g=d2, x=e1)
#         d1 = torch.cat([a2, d2], dim=1)

#         d0 = self.up_conv2(self.conv2(d1))
#         final = self.up_conv1(self.conv1(d0))

#         return final


# class Unet_DecoderEfficientNet(nn.Module):
#     """
#         Decoder module for Unet
#     """
#     def __init__(self, encoder_filters):
#         super(Unet_DecoderResNet, self).__init__()
        
#         self.conv5 = conv_block(encoder_filters[3], 192)
#         self.up_conv5 = up_conv(192, 128)
#         self.att4 = Attention_block(
#             F_g=128, F_l=encoder_filters[2], F_int=encoder_filters[1])

#         self.conv4 = conv_block(encoder_filters[2] + 128, 128)
#         self.up_conv4 = up_conv(128, 96)
#         self.att3 = Attention_block(
#             F_g=96, F_l=encoder_filters[1], F_int=encoder_filters[0])

#         self.conv3 = conv_block(encoder_filters[1] + 96, 96)
#         self.up_conv3 = up_conv(96, 64)
#         self.att2 = Attention_block(
#             F_g=64, F_l=encoder_filters[0], F_int=encoder_filters[0])

#         self.conv2 = conv_block(encoder_filters[0] + 64, 64)
#         self.up_conv2 = up_conv(64, 48)

#         self.conv1 = conv_block(48, 48)
#         self.up_conv1 = up_conv(48, 32)

#         for filter in encoder_filters:

#             pass

        
#         init_weights(self)

#     def forward(self, encoder_output):
#         e1, e2, e3, e4 = encoder_output
#         d4 = self.up_conv5(self.conv5(e4))
#         d4 = F.interpolate(d4, size=(e3.size(2), e3.size(3)), mode="bilinear", align_corners=True)
#         a4 = self.att4(g=d4, x=e3)
#         d3 = torch.cat([a4, d4], dim=1)

#         d3 = self.up_conv4(self.conv4(d3))
#         d3 = F.interpolate(d3, size=(e2.size(2), e2.size(3)), mode="bilinear", align_corners=True)
#         a3 = self.att3(g=d3, x=e2)
#         d2 = torch.cat([a3, d3], dim=1)

#         d2 = self.up_conv3(self.conv3(d2))
#         d2 = F.interpolate(d2, size=(e1.size(2), e1.size(3)), mode="bilinear", align_corners=True)
#         a2 = self.att2(g=d2, x=e1)
#         d1 = torch.cat([a2, d2], dim=1)

#         d0 = self.up_conv2(self.conv2(d1))
#         final = self.up_conv1(self.conv1(d0))

#         return final

def test_1():
    from pytorch_toolbelt.utils import count_parameters

    model = resnet50_attunet(freeze_backbone=True)
    device = torch.device('cuda:0')

    a = torch.randn(1, 3, 1024, 1024, device=device)
    model = model.to(device)
    output = model(a) 

    test_model= False
    test_params=True

    if test_model:
        print(model)
        print('Module', model.modules) #it should contain encoder, decoder, and head
        print(output.shape) #it should be (1,1,1024,1024)

    if test_params:
        print(f'[INFO] total and trainable parameters in the model {count_parameters(model)}')
        encoder_params = filter(lambda p: p.requires_grad, model.encoder.parameters())
        decoder_params = filter(lambda p: p.requires_grad, model.decoder.parameters())
        if hasattr(model, 'head'):
            head_params = filter(lambda p: p.requires_grad, model.head.parameters())

        print(encoder_params, decoder_params, head_params)
def test_2():
    # encoder0 = E.B0Encoder(pretrained=True, layers=[0, 1, 2, 3, 4])    
    # print(encoder0)
    # print(encoder0.channels)
    # print('###################B2ENCODER#######################')
    # encoder2 = E.B2Encoder(pretrained=True, layers=[0,1,2,3,4])
    # print(encoder2)
    # print(encoder2.channels)
    print('###################B4ENCODER#######################')
    encoder4 = E.B4Encoder(pretrained=True, layers=[0,1,2,3,4])
    print(encoder4)
    print(encoder4.channels)
    encoder4 = encoder4.to(torch.device('cuda:0'))
    a = torch.randn(1, 3, 256, 256, device=torch.device('cuda:0'))
    output = encoder4(a)
    # print(output)
    for o in output:
        print(o.shape)
    # print(output.shape)

def test_3():
    encoder = Unet_Encoder()
    encoder = encoder.to('cuda:0')
    a = torch.randn(1, 3, 256, 256, device=torch.device('cuda:0'))
    output = encoder(a)
    for o in output:
        print(o.shape)

    print(encoder)
    print(encoder.filters)

def test_4():
    encoder = Unet_Encoder(backbone='mobilenetv3_large_100')
    att = Attention_Unet(1, 0.2, encoder)

    a = torch.randn(1, 3, 256, 256, device=torch.device('cuda:0'))
    att = att.to('cuda:0')
    print(att)
    print(att(a).shape)

if __name__ == '__main__':
    import timm
    model_lists = timm.list_models(filter='mobile**', pretrained=True)
    print(model_lists)
    encoder = timm.create_model('mobilenetv3_large_100', features_only=True, pretrained=True)
    a = torch.randn(1, 3, 256, 256)
    output = encoder(a)
    for o in output:
        print(o.shape)
    print(encoder.feature_info.channels())
    test_4()
