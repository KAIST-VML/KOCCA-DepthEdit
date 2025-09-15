
from lib2to3.pytree import Base
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.networks import layer
from models.networks import encoder

# From SPADE
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.normalization import SPADE

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Upsample, self).__init__()
        self.module = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.PReLU(num_parameters=out_channels, init=0.25),
        )

    def forward(self, x):
        return self.module(x)

class BaseDecoder(nn.Module):
    def __init__(self, encoder_type='MobileNetV2'):
        super(BaseDecoder, self).__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'MobileNetV2':
            self.skip_channels =  [1280, 64, 32, 24, 32, 16, 1]
            self.idx = [8,4,3,2,0]
        elif encoder_type in ['ResNet101', 'MiDaS']:
            self.skip_channels = [2048, 1024, 512, 256, 64, 8, 1]
            self.idx = [4,3,2,1,0]

class MonodepthSpadeDecoder4(BaseDecoder):
    '''STROKE 가이드를 빡세게 주는 디코더'''
    def __init__(self, opt, **kwargs):
        super(MonodepthSpadeDecoder4,self).__init__(**kwargs)
        print("(decoder) Create MonoSpadeDecoder with additional stroke guidance")

        channels = self.skip_channels

        self.up1 = Upsample(channels[0], channels[1])   # 14
        self.up2 = Upsample(channels[1]*3, channels[2]) # 28
        self.up3 = Upsample(channels[2]*3, channels[3]) # 56
        self.up4 = Upsample(channels[3]*1, channels[4])

        self.up5 = Upsample(channels[4]*3, channels[5]) # 224
        self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
        self.act = torch.nn.LeakyReLU()
        self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
        
        self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                channels[1], opt.semantic_nc)
        self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                channels[3], opt.semantic_nc)
        self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                channels[5], opt.semantic_nc)

    def forward(self, features, guide_features, seg): 
        idx = self.idx

        #seg_feature = self.seg_conv(seg_feature[0])
        x = self.up1(features[idx[0]]) # C:64
        
        '''spade normalization'''
        x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
      
        x = torch.cat((x, features[idx[1]], guide_features[-1]), dim=1) # C:64*3
        x = self.up2(x) # C:32
        x = torch.cat((x, features[idx[2]], guide_features[-2]), dim=1) # C:32*3
        x = self.up3(x) # C:24
        
        '''spade normalization'''
        x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
        
        x = self.up4(x) # C:32]

        x = torch.cat((x, features[idx[4]], guide_features[-4]), dim =1)
        x = self.up5(x)

        '''spade normalization'''
        x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False,**kwargs):
        super(MonodepthSpadeDecoder,self).__init__(**kwargs)
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:
            #self.seg_conv = nn.Conv2d(150, 24, kernel_size=1, stride=1, padding=0) #1/4 size of original
            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*2, channels[4])
    
            self.up5 = Upsample(channels[4]*2, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            

            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

        else:
            channels = [1280, 64, 32, 24, 32, 16, 1]
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*2, channels[4]) # 112
            self.up5 = Upsample(channels[4]*2, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)

    def forward(self, features, seg_feature=None, seg_edge=None):
        idx = self.idx
        seg = seg_edge

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            # '''spade normalization'''
            # x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            # '''-------------------'''
            x = self.up2(x) # C:32
            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[3]]), dim=1) # C:24*3
            # '''spade normalization'''
            # x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
            # '''-------------------'''

            x = self.up4(x) # C:32
            x = torch.cat((x, features[idx[4]]), dim=1) # C:32*2
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeAttnDecoder(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeAttnDecoder,self).__init__(**kwargs)
        print("(decoder) Initialize SPADE+Attention Decoder")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.att1 = layer.Attention(channels[1]*2, False)
            self.att2 = layer.Attention(channels[2]*2, False)

            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3], channels[4])
    
            self.up5 = Upsample(channels[4], channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)

            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)
    
    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            x = self.att1(x)
            # '''spade normalization'''
            # x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            # '''-------------------'''
            x = self.up2(x) # C:32
            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.att2(x)
            x = self.up3(x) # C:24
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
            x = self.up4(x) # C:32
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder5(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeDecoder5,self).__init__(**kwargs)
        print("(decoder) Create MonoSpadeDecoder without last skip")
        self.opt = opt

        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*1, channels[4])
    
            self.up5 = Upsample(channels[4]*1, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            
            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            # '''spade normalization'''
            # x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            # '''-------------------'''
            x = self.up2(x) # C:32
            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
            # x = torch.cat((x, features[idx[3]]), dim=1)
            x = self.up4(x) # C:32
            # x = torch.cat((x, features[idx[4]]), dim=1)
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            
        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder3(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeDecoder3,self).__init__(**kwargs)
        print("(decoder) Create MonoSpadeDecoder without last skip")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*1, channels[4])
    
            self.up5 = Upsample(channels[4]*1, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            
            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            # '''spade normalization'''
            # x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            # '''-------------------'''
            x = self.up2(x) # C:32
            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
            # x = torch.cat((x, features[idx[3]]), dim=1)
            x = self.up4(x) # C:32
            # x = torch.cat((x, features[idx[4]]), dim=1)
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder3_2(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeDecoder3_2,self).__init__(**kwargs)
        print("(decoder) Create MonoSpadeDecoder without last skip")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            channels[5] = 32
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*1, channels[4])
    
            self.up5 = Upsample(channels[4]*1, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.ReLU(True)
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            
            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            # '''spade normalization'''
            # x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            # '''-------------------'''
            x = self.up2(x) # C:32
            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
            # x = torch.cat((x, features[idx[3]]), dim=1)
            x = self.up4(x) # C:32
            # x = torch.cat((x, features[idx[4]]), dim=1)
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.act(self.last(self.act(self.last_(x))))

class MonodepthSpadeDecoder3_3(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeDecoder3_3,self).__init__(**kwargs)
        print("(decoder) Create MonoSpadeDecoder3_3 without last skip")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*1, channels[4])
    
            self.up5 = Upsample(channels[4]*1, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            
            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

            self.att1 = layer.Attention(channels[1])
            self.att2 = layer.Attention(channels[2])
            self.att3 = layer.Attention(channels[3])

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            x = self.att1(x)
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            x = self.up2(x) # C:32
            x = self.att2(x)

            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            x = self.att3(x)
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
           
            x = self.up4(x) # C:32
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder3_3_fullskip(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeDecoder3_3_fullskip,self).__init__(**kwargs)
        print("(decoder) Create MonodepthSpadeDecoder3_3_fullskip")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*2, channels[4])
    
            self.up5 = Upsample(channels[4]*2, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            
            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

            self.att1 = layer.Attention(channels[1])
            self.att2 = layer.Attention(channels[2])
            self.att3 = layer.Attention(channels[3])

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            x = self.att1(x)
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            x = self.up2(x) # C:32
            x = self.att2(x)

            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            x = self.att3(x)
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
            
            x = torch.cat((x, features[idx[3]]), dim=1)
            x = self.up4(x) # C:32

            x = torch.cat((x, features[idx[4]]), dim=1)
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder3_3_0(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeDecoder3_3_0,self).__init__(**kwargs)
        print("(decoder) Create MonoSpadeDecoder3_3_0 without last skip")
        print("(decoder) No SPADE blocks")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*1, channels[4])
    
            self.up5 = Upsample(channels[4]*1, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)

            self.att1 = layer.Attention(channels[1])
            self.att2 = layer.Attention(channels[2])
            self.att3 = layer.Attention(channels[3])

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            x = self.att1(x)
            
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            x = self.up2(x) # C:32
            x = self.att2(x)

            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            x = self.att3(x)
           
            x = self.up4(x) # C:32
            x = self.up5(x)

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder3_3_1(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeDecoder3_3_1,self).__init__(**kwargs)
        print("(decoder) Create MonoSpadeDecoder3_3_1 without last skip")
        print("(decoder) Attention block after concat")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*1, channels[4])
    
            self.up5 = Upsample(channels[4]*1, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            
            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

            self.att1 = layer.Attention(channels[1]*2)
            self.att2 = layer.Attention(channels[2]*2)

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            x = self.att1(x)
            x = self.up2(x) # C:32

            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.att2(x)
            x = self.up3(x) # C:24

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
           
            x = self.up4(x) # C:32
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder3_4(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeDecoder3_4,self).__init__(**kwargs)
        print("(decoder) Create MonoSpadeDecoder3_4 without last skip")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*1, channels[4])
    
            self.up5 = Upsample(channels[4]*1, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            
            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

            self.att1 = layer.Attention(channels[1])
            self.att2 = layer.Attention(channels[2])
            self.att3 = layer.Attention(channels[3])
            self.att4 = layer.Attention(channels[4])

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            x = self.att1(x)
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            x = self.up2(x) # C:32
            x = self.att2(x)

            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            x = self.att3(x)
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
           
            x = self.up4(x) # C:32
            x = self.att4(x)
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder3_5(BaseDecoder):
    #5 SPADE block
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthSpadeDecoder3_5,self).__init__(**kwargs)
        print("(decoder) Create MonoSpadeDecoder3_5 without last skip")
        print("(dcoder) SPADE block at every level")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*1, channels[4])
    
            self.up5 = Upsample(channels[4]*1, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            
            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm2 = SPADE('spadesyncbatch3x3', \
                 channels[2], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm4 = SPADE('spadesyncbatch3x3', \
                 channels[4], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

            self.att1 = layer.Attention(channels[1])
            self.att2 = layer.Attention(channels[2])
            self.att3 = layer.Attention(channels[3])

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            x = self.up1(features[idx[0]]) # C:64
            x = self.att1(x)
            '''spade normalization 1'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            x = self.up2(x) # C:32
            x = self.att2(x)
            '''spade normalization 2'''
            x = F.leaky_relu(self.spade_norm2(x, seg), 2e-1)

            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            x = self.att3(x)
            '''spade normalization 3'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
           
            x = self.up4(x) # C:32
            '''spade normalization 4'''
            x = F.leaky_relu(self.spade_norm4(x, seg), 2e-1)

            x = self.up5(x)
            '''spade normalization 5'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.last(self.act(self.last_(x)))

class MonodepthSpadeDecoder2(nn.Module):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False):
        super(MonodepthSpadeDecoder2,self).__init__()
        print("(decoder) Create MonoSpadeDecoder without last skip")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = [1280, 64, 32, 24, 32, 16, 1]
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*2, channels[4])
    
            self.up5 = Upsample(channels[4], channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)

            self.spade_norm1 = SPADE('spadesyncbatch3x3', \
                 channels[1], opt.semantic_nc)
            self.spade_norm3 = SPADE('spadesyncbatch3x3', \
                 channels[3], opt.semantic_nc)
            self.spade_norm5 = SPADE('spadesyncbatch3x3', \
                 channels[5], opt.semantic_nc)

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[8]) # C:64
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[4]), dim=1) # C:64*2
            # '''spade normalization'''
            # x = F.leaky_relu(self.spade_norm1(x, seg), 2e-1) # C: 24*3
            # '''-------------------'''
            x = self.up2(x) # C:32
            x = torch.cat((x, features[3]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
            '''-------------------'''
            x = torch.cat((x, features[2]), dim=1) # C:24*3
            # '''spade normalization'''
            # x = F.leaky_relu(self.spade_norm3(x, seg), 2e-1) # C: 24*3
            # '''-------------------'''

            x = self.up4(x) # C:32
            x = self.up5(x)

            '''spade normalization'''
            x = F.leaky_relu(self.spade_norm5(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

        return self.last(self.act(self.last_(x)))

class MonodepthSFDecoder(nn.Module):
    '''
    Decoder that takes semantic feature input
    '''
    def __init__(self, seg_guide=False, sum_feature=False, seg_edge=False):
        super(MonodepthSFDecoder,self).__init__()
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:
            self.seg_conv = nn.Conv2d(150, 24, kernel_size=1, stride=1, padding=0) #1/4 size of original
            
            channels = [1280, 64, 32, 24, 32, 16, 1]
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            
            if self.sum_feature:
                self.up4 = Upsample(channels[3]*2, channels[4])
            else:
                if seg_edge:
                    self.up4 = Upsample(channels[3]*4, channels[4])
                else:
                    self.up4 = Upsample(channels[3]*3, channels[4])
                
            self.up5 = Upsample(channels[4]*2, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)

    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_feature[0]
        if self.seg_guide:
            seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[8]) # C:64
            x = torch.cat((x, features[4]), dim=1) # C:64*2
            x = self.up2(x) # C:32
            x = torch.cat((x, features[3]), dim=1) # C:32*2
            x = self.up3(x) # C:24

            if self.sum_feature: #sum resulting feature from previous scale to seg_feature
                x = torch.cat((x + seg_feature, features[2]), dim=1)
            else: 
                if seg_edge is not None:
                    x = torch.cat((x, features[2], seg_feature, seg_edge), dim=1) # C:24*4
                else:
                    x = torch.cat((x, features[2], seg_feature), dim=1) # C:24*3


            x = self.up4(x) # C:32
            x = torch.cat((x, features[0]), dim=1) # C:32*2
            x = self.up5(x)

        return self.last(self.act(self.last_(x)))

class MonodepthBaseDecoder(BaseDecoder):
    def __init__(self, opt, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthBaseDecoder,self).__init__(**kwargs)
        print("(decoder) Create Base Decoder")
        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:            
            channels = self.skip_channels
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*1, channels[4])
    
            self.up5 = Upsample(channels[4]*1, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)


    def forward(self, features, seg_feature=None, seg_edge=None):
        seg = seg_edge
        idx = self.idx

        if self.seg_guide:
            #seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[idx[0]]) # C:64
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            x = self.up2(x) # C:32
            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.up3(x) # C:24
            x = self.up4(x) # C:32
            x = self.up5(x)

        return self.last(self.act(self.last_(x)))

class MonodepthDecoder(BaseDecoder):
    def __init__(self, seg_guide=False, sum_feature=False, seg_edge=False, **kwargs):
        super(MonodepthDecoder,self).__init__(**kwargs)
        print("(decoder)    Initialize Depth Decoder")

        self.seg_guide = seg_guide
        self.sum_feature = sum_feature
        if self.seg_guide:
            self.seg_conv = nn.Conv2d(150, 24, kernel_size=1, stride=1, padding=0) #1/4 size of original
            
            channels = [1280, 64, 32, 24, 32, 16, 1]
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            
            if self.sum_feature:
                self.up4 = Upsample(channels[3]*2, channels[4])
            else:
                if seg_edge:
                    self.up4 = Upsample(channels[3]*4, channels[4])
                else:
                    self.up4 = Upsample(channels[3]*3, channels[4])
                
            self.up5 = Upsample(channels[4]*2, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)
            
            self.spade_norm = SPADE('spadesyncbatch3x3', \
                 channels[3]*3, 150)

        else:
            channels = self.skip_channels
            
            self.att1 = layer.Attention(channels[1]*2)
            self.att2 = layer.Attention(channels[2]*2)
            self.att3 = layer.Attention(channels[3]*2)
            # channels = [1280, 320, 160, 96, 64, 32, 24, 32, 3]
            
            self.up1 = Upsample(channels[0], channels[1])   # 14
            self.up2 = Upsample(channels[1]*2, channels[2]) # 28
            self.up3 = Upsample(channels[2]*2, channels[3]) # 56
            self.up4 = Upsample(channels[3]*2, channels[4]) # 112
            self.up5 = Upsample(channels[4]*2, channels[5]) # 224
            self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
            self.act = torch.nn.LeakyReLU()
            self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)

    def forward(self, features, seg_feature=None, seg_edge=None):

        idx = self.idx
        
        if self.seg_guide:
            seg = seg_feature[0]
            seg_feature = self.seg_conv(seg_feature[0])
            x = self.up1(features[8]) # C:64
            x = torch.cat((x, features[4]), dim=1) # C:64*2
            x = self.up2(x) # C:32
            x = torch.cat((x, features[3]), dim=1) # C:32*2
            x = self.up3(x) # C:24

            if self.sum_feature: #sum resulting feature from previous scale to seg_feature
                x = torch.cat((x + seg_feature, features[2]), dim=1)
            else: 
                if seg_edge is not None:
                    x = torch.cat((x, features[2], seg_feature, seg_edge), dim=1) # C:24*4
                else:
                    x = torch.cat((x, features[2], seg_feature), dim=1) # C:24*3
            
            '''spade normalization'''
            # x = F.leaky_relu(self.spade_norm(x, seg), 2e-1) # C: 24*3
            '''-------------------'''

            x = self.up4(x) # C:32
            x = torch.cat((x, features[0]), dim=1) # C:32*2
            x = self.up5(x)
        else:
            x = self.up1(features[idx[0]]) # C:64
            x = torch.cat((x, features[idx[1]]), dim=1) # C:64*2
            x = self.att1(x)
            x = self.up2(x) # C:32
            x = torch.cat((x, features[idx[2]]), dim=1) # C:32*2
            x = self.att2(x)
            x = self.up3(x) # C:24
            x = torch.cat((x, features[idx[3]]), dim=1) # C:24*2
            x = self.att3(x)
            x = self.up4(x) # C:32
            x = torch.cat((x, features[idx[4]]), dim=1) # C:32*2
            x = self.up5(x)

        return self.last(self.act(self.last_(x)))

class MultiHeadDecoder(nn.Module):
    def __init__(self, num_class=150, use_softmax=False):
        super(MultiHeadDecoder,self).__init__()
        self.use_softmax=use_softmax

        channels = [1280, 64, 32, 24, 32, 16, 1]
        self.up1 = Upsample(channels[0], channels[1])   # 14
        self.up2 = Upsample(channels[1]*2, channels[2]) # 28
        self.up3 = Upsample(channels[2]*2, channels[3]) # 56
        self.up4 = Upsample(channels[3]*2, channels[4]) # 112
        self.up5 = Upsample(channels[4]*2, channels[5]) # 224
        self.last_ = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
        self.act = torch.nn.LeakyReLU()
        self.last = nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1)

        fc_dim = channels[3]
        self.cbr = layer.conv3x3_bn_relu(fc_dim, fc_dim //4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, features, segSize=None):
        x = self.up1(features[8]) # C:64
        x = torch.cat((x, features[4]), dim=1) # C:64*2
        x = self.up2(x) # C:32
        x = torch.cat((x, features[3]), dim=1) # C:32*2
        x = self.up3(x) # C:24
        ''' Two head: Depth'''
        x1 = torch.cat((x, features[2]), dim=1) # C:24*2
        x1 = self.up4(x1) # C:32
        x1 = torch.cat((x1, features[0]), dim=1) # C:32*2
        x1 = self.up5(x1)
        x1 = self.last(self.act(self.last_(x1)))
        '''Two head: Segmentation'''
        x2 = self.cbr(x)
        x2 = self.conv_last(x2)
        if self.use_softmax: # is True during inference
            x2 = nn.functional.interpolate(
                x2, size=segSize, mode='bilinear', align_corners=False)
            x2 = nn.functional.softmax(x2, dim=1)
        else:
            x2 = nn.functional.interpolate(
                x2, size=segSize, mode='bilinear', align_corners=False)
            x2 = nn.functional.log_softmax(x2, dim=1)

        return x1, x2

class SegDecoder(nn.Module):
    def __init__(self, num_class=150, use_softmax=False):
        super(SegDecoder,self).__init__()
        self.use_softmax=use_softmax

        channels = [1280, 64, 32, 24, 32, 16, 1]
        self.up1 = Upsample(channels[0], channels[1])   # 14
        self.up2 = Upsample(channels[1]*2, channels[2]) # 28
        self.up3 = Upsample(channels[2]*2, channels[3]) # 56

        #if using attention
        self.att1 = layer.Attention(channels[1]*2)
        self.att2 = layer.Attention(channels[2]*2)

        fc_dim = channels[3]
        self.cbr = layer.conv3x3_bn_relu(fc_dim, fc_dim //4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, features, segSize=None):
        x = self.up1(features[8]) # C:64
        x = torch.cat((x, features[4]), dim=1) # C:64*2
        x = self.att1(x)
        x = self.up2(x) # C:32
        x = torch.cat((x, features[3]), dim=1) # C:32*2
        x = self.att2(x)
        x = self.up3(x) # C:24
       
        '''Segmentation'''
        x = self.cbr(x)
        x = self.conv_last(x)
        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.log_softmax(x, dim=1)

        return x


class SPADEDecoder(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        print("(decoder) Initialize Original SPADE Decoder")
        self.opt = opt
        nf = opt.ngf

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        #self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.conv_depth = nn.Conv2d(final_nc,1, 3, padding=1)
        
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, seg):
        '''
        x: encoded feature
        seg: segmap

        '''
        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_depth(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        
        #return x
        return torch.nn.functional.threshold(input=x, threshold=0.0, value=0.0)        


class HRNetDecoder(nn.Module):
    
    def __init__(self, num_class=150, fc_dim=720, use_softmax=False):
        super(HRNetDecoder,self).__init__()
        self.use_softmax = use_softmax

        self.cbr = layer.conv3x3_bn_relu(fc_dim, fc_dim //4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        decoded_features = []

        conv5 = conv_out[-1]
        x = self.cbr(conv5); decoded_features.append(x)
        x = self.conv_last(x); decoded_features.append(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        decoded_features.append(x)

        return x, decoded_features