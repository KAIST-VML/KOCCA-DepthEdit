"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock

import models.networks.encoder as encoders
import models.networks.decoder as decoders

class MultiHeadDepthGenerator(BaseNetwork):
    '''
    Generate depth and label map at the same time
    '''
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.encoder = encoders.MobileNetV2Encoder_Skip(pretrained=True, num_channels=self.opt.input_ch)
        # self.decoder = decoders.MultiHeadDecoder(num_class=opt.semantic_nc)
        self.decoder_depth = decoders.MonodepthDecoder()
        self.decoder_seg = decoders.SegDecoder(num_class=opt.semantic_nc)

        if self.opt.input_ch == 3:
            if self.opt.refine_depth:
                self.conv = nn.Conv2d(5, 3, 3, padding=1)
            else:
                self.conv = nn.Conv2d(4, 3, 3, padding=1)

    def forward(self, input):
        _,_,self.sh,self.sw = input.shape
        seg_size = (self.sh, self.sw)
        
        if self.opt.segmap_type == "": # segmap_type not given
            x = input
            seg = None
        else:
            x = input[:,:4]
            seg = input[:,4:]

        if self.opt.input_ch == 3 and input.shape[1] > 3: #has guide +alpha
            x = self.conv(x)
        
        features = self.encoder(x)
        # disp, new_label = self.decoder(features, seg_size)
        disp = self.decoder_depth(features)
        # new_label = None
        new_label = self.decoder_seg(features, seg_size)
        return torch.nn.functional.threshold(input=disp, threshold=0.0, value=0.0), \
            new_label
    

    def postprocess(self, feature, scale_factor):
        _, pred_n = torch.max(feature, dim = 1) # Bx1xHxW
        out = pred_n.type(torch.cuda.FloatTensor).unsqueeze(1)
        return out


class SegDepthGenerator(BaseNetwork):
    def __init__(self, opt, weights=None):
        super().__init__()
        self.opt = opt

        if self.opt.encoder == 'MobileNetV2':
            self.encoder = encoders.MobileNetV2Encoder_Skip(pretrained=True, num_channels=self.opt.input_ch)
        elif self.opt.encoder == 'MiDaS':
            if self.opt.full_skip:
                self.encoder = encoders.MiDaSEncoder_FullSkip(pretrained=True, num_channels=self.opt.input_ch)
            else:
                self.encoder = encoders.MiDaSEncoder_Skip(pretrained=True, num_channels=self.opt.input_ch)
        elif self.opt.encoder == 'ResNet101':
            self.encoder = encoders.ResNetEncoder_Skip(pretrained=True, num_channels=self.opt.input_ch)
        '''20211115'''
        # self.decoder = decoders.MonodepthDecoder(seg_guide=True)
        '''20211116 no skip at 4 and 5'''
        if opt.use_attention:
            '''20220216 attention module'''
            self.decoder = decoders.MonodepthSpadeAttnDecoder(self.opt, seg_guide=True, sum_feature=False, seg_edge=None, encoder_type=self.opt.encoder)
        if self.opt.decoder == 'spade':
            ''' Spade Module'''
            opt.norm_G = 'spectralspadesyncbatch3x3'
            opt.num_upsampling_layers = 'normal'
            self.decoder = decoders.SPADEDecoder(opt)
            if self.opt.encoder == 'MiDaS':
                self.conv_ch = nn.Conv2d(2048, 1024, 3, padding=1)
            else:
                self.conv_ch = nn.Conv2d(1280, 1024, 3, padding=1)
        elif self.opt.guide_encoder:
            self.guide_encoder = encoders.StrokeEncoder(encoder_type=self.opt.encoder)
            self.decoder = decoders.MonodepthSpadeDecoder4(self.opt, encoder_type=self.opt.encoder)    
        else:
            if self.opt.decoder == 'base':
                self.decoder = decoders.MonodepthBaseDecoder(self.opt, seg_guide=True, sum_feature=False, \
                                                            seg_edge=None, encoder_type=self.opt.encoder)
            elif self.opt.test_decoder: #baseline
                if self.opt.decoder == 'default' or self.opt.decoder == "s3":
                    self.decoder = decoders.MonodepthSpadeDecoder3_3(self.opt, seg_guide=True, sum_feature=False, \
                                                                seg_edge=None, encoder_type=self.opt.encoder)
                elif self.opt.decoder == 's0':
                    self.decoder = decoders.MonodepthSpadeDecoder3_3_0(self.opt, seg_guide=True, sum_feature=False, \
                                                                seg_edge=None, encoder_type=self.opt.encoder)
                elif self.opt.decoder == 'a0':
                    self.decoder = decoders.MonodepthSpadeDecoder3(self.opt, seg_guide=True, sum_feature=False, \
                                                            seg_edge=None, encoder_type=self.opt.encoder)
                elif self.opt.decoder == 's5':
                    self.decoder = decoders.MonodepthSpadeDecoder3_5(self.opt, seg_guide=True, sum_feature=False, \
                                                            seg_edge=None, encoder_type=self.opt.encoder)
            else:
                self.decoder = decoders.MonodepthSpadeDecoder3(self.opt, seg_guide=True, sum_feature=False, \
                                                            seg_edge=None, encoder_type=self.opt.encoder)
        # if  opt.segmap_type == "": #if segmap_type is not given
        #     self.seg_encoder = encoders.HRNetEncoder()
        #     self.seg_decoder = decoders.HRNetDecoder()
        #     if weights is not None:
        #         self.seg_decoder.load_state_dict(
        #             torch.load(weights, map_location=lambda storage, loc: storage), strict=False
        #         )
        #     '''freeze params of segmentation modules'''
        #     for param in self.seg_encoder.parameters(): #freeze encoder
        #         param.requires_grad = False
        #     for param in self.seg_decoder.parameters(): #freeze decoder 
        #         param.requires_grad = False

        if not self.opt.no_guide and self.opt.input_ch == 3:
            if self.opt.refine_depth:
                self.conv = nn.Conv2d(5,3,3,padding=1) #5 input channetl (rgb + guide + incomplete disp)
            else:
                self.conv = nn.Conv2d(4, 3, 3, padding=1)



    def forward(self, input):
        '''process input'''
        _,_,self.sh,self.sw = input.shape
        self.sh = self.sh//4
        self.sw = self.sw//4

        if self.opt.input_ch == 3:
            if not self.opt.no_guide:
                x = self.conv(input[:,:4])
            else:
                x = input[:,:3]
        else:
            x = input[:,:self.opt.input_ch]

        if self.opt.decoder == "spade" or self.opt.segmap_type == "coco-stuff" or self.opt.segmap_type == "ade20k":
            seg = input[:,self.opt.input_ch:]
        else:
            seg = None

        '''feed input to the network'''
        depth_feature = self.encoder(x)
        new_label = None
        if self.opt.decoder == "spade":
            disp = self.decoder(self.conv_ch(depth_feature[-1]), seg)
        else:
            if self.opt.segmap_type == "":
                ''' if semantic feature is given '''
                seg_feature = self.seg_encoder(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))(input[:,:3,:,:]))
                seg_decoded_feature = self.seg_decoder(seg_feature) # (final feature, all the intermediate features)
                disp = self.decoder(depth_feature, seg_decoded_feature, seg)
                new_label = self.postprocess(seg_decoded_feature[0], scale_factor = 4)
            elif self.opt.guide_encoder:
                guide_features = self.guide_encoder(input[:,3].unsqueeze(1))
                disp = self.decoder(depth_feature, guide_features, seg)
            else:
                ''' if complete segmap is given '''
                disp = self.decoder(depth_feature, None, seg)
                

        return torch.nn.functional.threshold(input=disp, threshold=0.0, value=0.0), \
            new_label
    
    def postprocess(self, feature, scale_factor):
        _, pred_n = torch.max(feature, dim = 1) # Bx1xHxW
        out = nn.functional.interpolate(pred_n.type(torch.cuda.FloatTensor).unsqueeze(1), scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return out

class DepthGenerator(BaseNetwork):
    def __init__(self,opt):
        super().__init__()
        self.opt = opt
        self.input_ch = self.opt.input_ch

        if self.opt.encoder == 'MobileNetV2':
            self.encoder = encoders.MobileNetV2Encoder_Skip(pretrained=True, num_channels=self.opt.input_ch)
        elif self.opt.encoder == 'MiDaS':
            self.encoder = encoders.MiDaSEncoder_Skip(pretrained=True, num_channels=self.opt.input_ch)
        # self.encoder = encoders.MobileNetV2Encoder_Skip(pretrained=True, num_channels=self.input_ch)
        self.decoder = decoders.MonodepthDecoder(encoder_type=self.opt.encoder)
        
        if self.input_ch == 3:
            if self.opt.refine_depth:
                self.conv = nn.Conv2d(5, 3, 3, padding=1)
            else:
                self.conv = nn.Conv2d(4, 3, 3, padding=1)
        
    def forward(self, input):
        # x = input[:,:3,:,:]
        # guide = input[:,-1,:,:]
        
        if self.input_ch == 3 and input.shape[1] > 3: #has guide +alpha
            x = self.conv(input)
        else:
            x = input
        
        '''encode feature'''
        feat = self.encoder(x)

        '''decode feature'''
        disp = self.decoder(feat)        

        # feat = self.encoder(input)
        # disp = self.decoder(feat)

        return torch.nn.functional.threshold(input=disp, threshold=0.0, value=0.0), None

class Seg2DepthGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        elif (self.opt.mask != "" or self.opt.use_rgb) and not self.opt.no_label_encoder:
            # When using something more than semantic map
            input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) \
            + (0 if opt.no_instance else 1)

            if opt.mask != "": input_nc += 1
            if opt.use_rgb: input_nc += 3

            self.fc = self.get_pix2pixHD_encoder(opt, input_nc)
        elif self.opt.mask != "" and self.opt.no_label_encoder:
            self.fc1 = self.get_pix2pixHD_encoder(opt, 1) # masked depth 1 channel
            self.fc2 = nn.Conv2d(self.opt.semantic_nc-1, 16 * nf, 3, padding=1)
            self.fc3 = nn.Conv2d(16*nf*2, 16*nf, 1, padding = 0)
            print('!!!!!!!1111!!!!!!!!!!!')
        elif self.opt.use_rgb and self.opt.no_label_encoder:
            # When using rgb image and encode it separately from semantic labels
            if self.opt.mobilenet:
                self.fc1 = MobileNetV2Encoder()
            else:
                self.fc1 = self.get_pix2pixHD_encoder(opt, 3) # rgb 3 channel
            self.fc2 = nn.Conv2d(self.opt.semantic_nc-3, 16 * nf, 3, padding=1)
            self.fc3 = nn.Conv2d(16*nf*2, 16*nf, 1, padding = 0)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
            print('!!!!!!!2222222!!!!!!!!!!!')
            
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

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
    
        return sw, sh

    def get_pix2pixHD_encoder(self, opt, input_nc):
        nf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, 'instance')
        activation = nn.ReLU(False)

        model = []
        opt.resnet_initial_kernel_size = 7
        opt.resnet_n_downsample = 4

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                norm_layer(nn.Conv2d(input_nc, nf,
                                    kernel_size=opt.resnet_initial_kernel_size,
                                    stride=2, padding=0)),
                activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                        kernel_size=3, stride=2, padding=1)),
                    activation]
            mult *= 2

        return nn.Sequential(*model)

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        
        elif (self.opt.mask != "" or self.opt.use_rgb) and not self.opt.no_label_encoder:
            x = self.fc(seg)

        elif self.opt.mask != "" and self.opt.no_label_encoder:
            ''' Put both masked depth and segmap together '''
            # masked_image = seg[:,:1,:,:]
            # segmaps = seg[:,1:,:,:]

            # x1 = self.fc1(masked_image)
            
            # x2 = F.interpolate(segmaps, size = (self.sh, self.sw))
            # x2 = self.fc2(x2)
            
            # x = torch.cat([x1,x2], dim = 1)
            # x = self.fc3(x)

            ''' Put only masked depth '''
            masked_image = seg[:,:1,:,:]
            seg = seg[:,1:,:,:]
            x = self.fc1(masked_image)

        elif self.opt.use_rgb and self.opt.no_label_encoder:
            ''' Put both rgb and segmap together '''

            # rgb = seg[:,:3,:,:]
            # segmaps = seg[:,3:,:,:]

            # x1 = self.fc1(rgb)
            
            # x2 = F.interpolate(segmaps, size = (self.sh, self.sw))
            # x2 = self.fc2(x2)

            # x = torch.cat([x1,x2], dim = 1)
            # x = self.fc3(x)
            ''' Put only rgb '''
            rgb = seg[:,:3,:,:]
            seg = seg[:,3:,:,:]
            x = self.fc1(rgb)

        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
        
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


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)
  
        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

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

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

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


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
