# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .Vit import VisionTransformer, Reconstruct,upSkip,gray_Cross_Attention
from .pixlevel import PixLevelModule
from .CSR import FeatureDetailDistiller


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU',reliNum=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.reliNum = reliNum
        self.pixModule = PixLevelModule(in_channels)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.nConvs2 = _make_nConv(in_channels * 2, out_channels, nb_Conv, activation)
        self.CSR = FeatureDetailDistiller(in_channels=in_channels, decode_channels=in_channels)

    def forward(self, x, skip_x,name=None):
        x = self.up(x)  
        skip_x_att = self.pixModule(skip_x) 
        if x.shape[1]!=64:
            up = self.CSR(x,skip_x_att)
            return self.nConvs(up)
        else:
            up = torch.cat([skip_x_att, x], dim=1) 
            return self.nConvs2(up)


class LViT(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 8, in_channels * 4, nb_Conv=2,reliNum=4)
        self.up3 = UpblockAttention(in_channels * 4, in_channels * 2, nb_Conv=2,reliNum=3)
        self.up2 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2,reliNum=2)
        self.up1 = UpblockAttention(in_channels, in_channels, nb_Conv=2,reliNum=1)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outc_cu2 = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        self.upSkip4 = upSkip(in_channels=512)
        self.cross_attention1 = gray_Cross_Attention(config,patch_size = 16,dim = 64,img_size=224,in_channels=64)
        self.cross_attention2 = gray_Cross_Attention(config,patch_size = 8,dim = 128,img_size=112,in_channels=128)
        self.cross_attention3 = gray_Cross_Attention(config,patch_size = 4,dim = 256,img_size=56,in_channels=256)
        self.cross_attention4 = gray_Cross_Attention(config,patch_size = 2,dim = 512,img_size=28,in_channels=512)

        self.conv_cat1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv_cat2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv_cat3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv_cat4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        

    def forward(self, x, text,name=''):
        x = x.float()  

        x1 = self.inc(x)  


        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2) 
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)

        x2 = self.down1(x1)
  
        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

       

        y1 = self.downVit(x1, x1, text1)

        y2 = self.downVit1(x2, y1, text2)

        y3 = self.downVit2(x3, y2, text3)
        
        y4 = self.downVit3(x4, y3, text4)
        

        average_cu2 = self.upSkip4(self.reconstruct4(y4))

        x1 = self.conv_cat1(torch.cat([x1, self.reconstruct1(y1)],dim=1))
        x2 = self.conv_cat2(torch.cat([x2, self.reconstruct2(y2)],dim=1))
        x3 = self.conv_cat3(torch.cat([x3, self.reconstruct3(y3)],dim=1))
        x4 = self.conv_cat4(torch.cat([x4, self.reconstruct4(y4)],dim=1))

        x = self.up4(x5, x4,name)
        x = self.up3(x, x3,name)
        x = self.up2(x, x2,name)
        x = self.up1(x, x1,name)
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
            average_cu2= self.last_activation(self.outc_cu2(average_cu2))
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
            average_cu2 = self.outc_cu2(average_cu2)

        return logits,average_cu2
