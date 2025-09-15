import torch
import torch.nn as nn
from models.networks.base_network import BaseNetwork

class MidasGenerator(BaseNetwork):
    '''
    Generate depth and label map at the same time
    '''
    def __init__(self, opt, pretrained=True, model_type='MiDaS'):
        super().__init__()
        print("(Generator) Create {} Generator".format(model_type))
        self.opt = opt
        num_channels = self.opt.input_ch
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        if num_channels > 3:
            conv1_weight = midas._modules['pretrained'].layer1[0].weight
            c_out, c_in, h, w = conv1_weight.size()
            conv1_weight_ = torch.zeros(c_out, num_channels, h,w)
            conv1_weight_[:,:3,:,:] = conv1_weight

            conv1 = nn.Conv2d(num_channels,c_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
            conv1.weight.data = nn.Parameter(conv1_weight_)
            midas._modules['pretrained'].layer1[0] = conv1

        self.model = midas
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def forward(self, input):
        if self.opt.input_ch == 3:
            x = input[:,:3]
        else:
            x = input[:,:4]
        # import pdb; pdb.set_trace()
        out = self.model(x).unsqueeze(1)
        return out, None

class DPTGenerator(BaseNetwork):
    '''
    DPT model with additional input channel for stroke
    '''
    def __init__(self, opt, pretrained=True, model_type='DPT_Large'):
        super().__init__()
        print("(Generator) Create {} Generator".format(model_type))
        self.opt = opt
        num_channels = self.opt.input_ch
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        if num_channels > 3:
            conv1_weight = midas._modules['pretrained'].model.patch_embed.proj.weight
            conv1_bias =  midas._modules['pretrained'].model.patch_embed.proj.bias
            c_out, c_in, h, w = conv1_weight.size()
            conv1_weight_ = torch.zeros(c_out, num_channels, h,w)
            conv1_weight_[:,:3,:,:] = conv1_weight

            conv1 = nn.Conv2d(num_channels,c_out,
                                kernel_size=(16, 16), 
                                stride=(16, 16), 
                                bias=True)

            conv1.weight.data = nn.Parameter(conv1_weight_)
            conv1.bias.data = nn.Parameter(conv1_bias)
            midas._modules['pretrained'].model.patch_embed.proj = conv1

        self.model = midas
        # self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def forward(self, input):
        if self.opt.input_ch == 3:
            x = input[:,:3]
        else:
            x = input[:,:4]
        # import pdb; pdb.set_trace()
        out = self.model(x).unsqueeze(1)
        return out, None