"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks import layer

import torchvision.models as models


def load_midas(model_type):
    import os
    if os.path.exists("../cache/torch/hub/"):
        return torch.hub.load("../cache/torch/hub/intel-isl_MiDaS_master/",model = model_type, source='local')
    else:
        return torch.hub.load("intel-isl/MiDaS", model_type)
        

class StrokeEncoder(nn.Module):
    def __init__(self, channels=None, encoder_type='MobileNetV2'):
        super(StrokeEncoder, self).__init__()
        print("(encoder)    Initializing Stroke Encoder")

        if not channels is None:
            self.channels = channels
        else:
            if encoder_type == 'MobileNetV2':
                self.channels = [1280, 64, 32, 24, 32, 1] 
                # feature size  [      14, 28, 56, 128,256]
            else:
                self.channels = [2048, 1024, 512, 256, 64, 8, 1]

        self.layers = []
        N = len(self.channels)
        for i in range(N-2):
            in_ch = self.channels[-1-i]
            out_ch = self.channels[-2-i]
            self.layers += [layer.conv3x3_bn_relu(in_ch, out_ch,stride=2)]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        self.guide_features = []

        for conv_layer in list(self.model.children()):
            x = conv_layer(x)
            self.guide_features.append(x)
        
        return self.guide_features
        

class MiDaSEncoder_Skip(nn.Module):
    def __init__(self, pretrained=True, num_channels=3, model_type='MiDaS'):
        super(MiDaSEncoder_Skip, self).__init__()
        print("(encoder)    Initialize MiDaS Encoder")
        # midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas = load_midas(model_type)
        if num_channels > 3:
            conv1_weight = midas._modules['pretrained'].layer1[0].weight
            c_out, c_in, h, w = conv1_weight.size()
            conv1_weight_ = torch.zeros(c_out, num_channels, h,w)
            conv1_weight_[:,:3,:,:] = conv1_weight

            conv1 = nn.Conv2d(num_channels,c_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            conv1.weight.data = nn.Parameter(conv1_weight_)
            midas._modules['pretrained'].layer1[0] = conv1

        modules = list(midas._modules['pretrained'].children())

        #say input is 256
        self.enc_1 = nn.Sequential(*modules[0][0:4])   # WH:128 # C:64   # 0
        self.enc_2 = nn.Sequential(*modules[0][4])    # WH:64 # C:256   # 1
        self.enc_3 = nn.Sequential(*modules[1])   # WH:32  # C:512   # 2
        self.enc_4 = nn.Sequential(*modules[2])   # WH:16  # C:1024   # 3
        self.enc_5 = nn.Sequential(*modules[3])  # WH:8  # C:2048   # 4
        # self.enc_6 = nn.Sequential(modules[8], nn.Flatten(1), modules[9]) # WH:1 # C:1000 #5

    def forward(self, input_image):
        self.features = []

        self.features.append(self.enc_1(input_image))
        self.features.append(self.enc_2(self.features[-1]))
        self.features.append(self.enc_3(self.features[-1]))
        self.features.append(self.enc_4(self.features[-1]))
        self.features.append(self.enc_5(self.features[-1]))
        # self.features.append(self.enc_6(self.features[-1]))
        
        if False:
            for idx, features in enumerate(self.features):
                print("[{}] {} ".format(idx,features.size()))
            
        return self.features

class MiDaSEncoder_FullSkip(nn.Module):
    def __init__(self, pretrained=True, num_channels=3, model_type='MiDaS'):
        super(MiDaSEncoder_FullSkip, self).__init__()
        print("(encoder)    Initialize MiDaS Encoder for Full Skip Connection")
        # midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas = load_midas(model_type)
        if num_channels > 3:
            conv1_weight = midas._modules['pretrained'].layer1[0].weight
            c_out, c_in, h, w = conv1_weight.size()
            conv1_weight_ = torch.zeros(c_out, num_channels, h,w)
            conv1_weight_[:,:3,:,:] = conv1_weight

            conv1 = nn.Conv2d(num_channels,c_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            conv1.weight.data = nn.Parameter(conv1_weight_)
            midas._modules['pretrained'].layer1[0] = conv1

        modules = list(midas._modules['pretrained'].children())

        #say input is 256
        self.enc_1 = nn.Sequential(*modules[0][0:3])   # WH:128 # C:64   # 0
        self.enc_2 = nn.Sequential(*modules[0][3:])    # WH:64 # C:256   # 1
        self.enc_3 = nn.Sequential(*modules[1])   # WH:32  # C:512   # 2
        self.enc_4 = nn.Sequential(*modules[2])   # WH:16  # C:1024   # 3
        self.enc_5 = nn.Sequential(*modules[3])  # WH:8  # C:2048   # 4
        # self.enc_6 = nn.Sequential(modules[8], nn.Flatten(1), modules[9]) # WH:1 # C:1000 #5

    def forward(self, input_image):
        self.features = []

        self.features.append(self.enc_1(input_image))
        self.features.append(self.enc_2(self.features[-1]))
        self.features.append(self.enc_3(self.features[-1]))
        self.features.append(self.enc_4(self.features[-1]))
        self.features.append(self.enc_5(self.features[-1]))
        # self.features.append(self.enc_6(self.features[-1]))
        
        if False:
            for idx, features in enumerate(self.features):
                print("[{}] {} ".format(idx,features.size()))
            
        return self.features              

class ResNetEncoder_Skip(nn.Module):
    def __init__(self, pretrained=True, num_channels=3):
        super(ResNetEncoder_Skip, self).__init__()

        print("(encoder)    Initializing ResNet101 Encoder")
        model = models.resnet101(pretrained=pretrained, progress=True)

        if num_channels > 3:
                print("(encoder)    Modifying input layer")

                conv1_weight = model.conv1._parameters['weight']
                c_out, c_in, h, w = conv1_weight.size()
                conv1_weight_ = torch.zeros(c_out, num_channels, h,w)
                conv1_weight_[:,:3,:,:] = conv1_weight

                conv1 = nn.Conv2d(num_channels,c_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                conv1.weight.data = nn.Parameter(conv1_weight_)
                model.conv1 = conv1
        modules = list(model.children())
        #say input is 256
        self.enc_1 = nn.Sequential(*modules[:4])   # WH:128 # C:64   # 0
        self.enc_2 = nn.Sequential(*modules[4])    # WH:64 # C:256   # 1
        self.enc_3 = nn.Sequential(*modules[5])   # WH:32  # C:512   # 2
        self.enc_4 = nn.Sequential(*modules[6])   # WH:16  # C:1024   # 3
        self.enc_5 = nn.Sequential(*modules[7])  # WH:8  # C:2048   # 4
        # self.enc_6 = nn.Sequential(modules[8], nn.Flatten(1), modules[9]) # WH:1 # C:1000 #5

    def forward(self, input_image):
        self.features = []
        self.features.append(self.enc_1(input_image))
        self.features.append(self.enc_2(self.features[-1]))
        self.features.append(self.enc_3(self.features[-1]))
        self.features.append(self.enc_4(self.features[-1]))
        self.features.append(self.enc_5(self.features[-1]))
        # self.features.append(self.enc_6(self.features[-1]))
        
        if False:
            for idx, features in enumerate(self.features):
                print("[{}] {} ".format(idx,features.size()))
            
        return self.features       
        

class MobileNetV2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Encoder, self).__init__()
        
        model = models.mobilenet_v2(pretrained=pretrained,progress=True)

        self.enc = nn.Sequential(*model.features[0:19])
        self.conv1d = nn.Conv2d(1280, 1024, 3, padding=1)

    def forward(self, input):
        return self.conv1d(self.enc(input))

class MobileNetV2Encoder_Skip(nn.Module):
    """ Pytorch module for a mobilenet v2 encoder
    """
    def __init__(self, pretrained=True, num_channels=3):
        super(MobileNetV2Encoder_Skip,self).__init__()

        model = models.mobilenet_v2(pretrained=pretrained,progress=True)
        
        if num_channels > 3:
            print("(encoder)    Modifying input layer")
            enc_init = model.features[0].state_dict()
            
            conv0_weights = enc_init['0.weight']
            c_out, c_in, h, w = conv0_weights.size()
            conv0_weights_ = torch.zeros(c_out, num_channels, h,w)
            conv0_weights_[:,:3,:,:] = conv0_weights

            conv0 = nn.Conv2d(num_channels,c_out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            conv0.weight.data = nn.Parameter(conv0_weights_)
            model.features[0]._modules['0'] = conv0


        self.enc_1 = nn.Sequential(*model.features[0:1])   # WH:112 # C:32   # 0
        self.enc_2 = nn.Sequential(*model.features[1:2])   # WH:112 # C:16   # 1
        self.enc_3 = nn.Sequential(*model.features[2:4])   # WH:56  # C:24   # 2
        self.enc_4 = nn.Sequential(*model.features[4:7])   # WH:28  # C:32   # 3
        self.enc_5 = nn.Sequential(*model.features[7:11])  # WH:14  # C:64   # 4
        self.enc_6 = nn.Sequential(*model.features[11:14]) # WH:14  # C:96 
        self.enc_7 = nn.Sequential(*model.features[14:17]) # WH:7   # C:160
        self.enc_8 = nn.Sequential(*model.features[17:18]) # WH:7   # C:320
        self.enc_9 = nn.Sequential(*model.features[18:19]) # WH:7   # C:1280 # 8


    def forward(self, input_image):
        self.features = []
        self.features.append(self.enc_1(input_image))
        self.features.append(self.enc_2(self.features[-1]))
        self.features.append(self.enc_3(self.features[-1]))
        self.features.append(self.enc_4(self.features[-1]))
        self.features.append(self.enc_5(self.features[-1]))
        self.features.append(self.enc_6(self.features[-1]))
        self.features.append(self.enc_7(self.features[-1]))
        self.features.append(self.enc_8(self.features[-1]))
        self.features.append(self.enc_9(self.features[-1]))

        if False:
            for idx, features in enumerate(self.features):
                print("[{}] {} ".format(idx,features.size()))
            
        return self.features
        

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

