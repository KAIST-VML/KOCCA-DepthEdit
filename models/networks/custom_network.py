import torch
import torch.nn as nn
import torch.nn.functional as F

class refineNet():
    def __init__(self, opt):
        self.opt = opt
        nf = opt.ngf

        model = []
        nc = 1
        up = nn.Upsample(scale_factor=2)
        activation = nn.LeakyReLU(2e-1)
        
        #downsample
        for i in range(iter):
            model += [nn.Conv2d(nc, nc * 2, kernel_size=3, stride=2, padding=1),
                    activation]
            nc *= 2
        
        #upsample
        for i in range(iter):

            model += [up, \
                    nn.Conv2d(int(nc), int(nc / 2), kernel_size=3, stride=1, padding=1), \
                    activation]
            nc /= 2

        self.refine = nn.Sequential(*model)

    def forward(self, input_image):
        x = self.refine(input_image)
        return nn.functional.threshold(input=x, threshold=0.0, value=0.0)

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

from models.networks.base_network import BaseNetwork
class refineNet2(BaseNetwork):
    def __init__(self, opt):
        super(refineNet2, self).__init__()
        self.opt = opt
        
        model = []
        nc = 1
        channels=[32,64,128]

        self.activation = nn.LeakyReLU(2e-1)

        self.down1 = nn.Sequential(*[nn.Conv2d(1, channels[0], kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(channels[0]),
                            self.activation
                            ])
        self.down2 = nn.Sequential(*[nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(channels[1]),
                            self.activation
                            ])
        self.down3 = nn.Sequential(*[nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1),
                           nn.BatchNorm2d(channels[2]),
                           self.activation
                            ])

        self.conv = nn.Sequential(*[nn.Conv2d(1, channels[2], kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(channels[2]),
                           nn.ReLU()
                            ])
        
        self.up1 = Upsample(channels[2]*2,channels[1])
        self.up2 = Upsample(channels[1]*2,channels[0])
        
        self.up3 = Upsample(channels[0]*2,8)

        self.last = nn.Sequential(
				nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
			)

        # self.eye = nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1)
        # self.eye = nn.parameter.Parameter(torch.ones((4,1,256,256)))

    def forward(self, input):
        '''
        input = depth + edge
        '''
        depth = input[:,0,:,:].unsqueeze(1)
        rgb = input[:,1:4,:,:]
        
        f1 = self.down1(rgb) # 128, 8
        f2 = self.down2(f1) # 64, 16
        f3 = self.down3(f2) # 32, 32

        d_conv = self.conv(depth) # h w 32
        
        x = torch.cat([f3,nn.functional.interpolate(d_conv, (f3.shape[2],f3.shape[3]), mode='bilinear', align_corners=False)],dim=1)
        x = self.up1(x)

        x = torch.cat([f2,x],dim=1)
        x = self.up2(x)

        x = torch.cat([f1,x],dim=1)
        x = self.up3(x)
        # x = self.eye + input
        return torch.nn.functional.threshold(input=self.last(x), threshold=0.0, value=0.0)

'''Kenburn'''
class Basic(torch.nn.Module):
	def __init__(self, strType, intChannels):
		super().__init__()

		if strType == 'relu-conv-relu-conv':
			self.netMain = torch.nn.Sequential(
				torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		elif strType == 'conv-relu-conv':
			self.netMain = torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		if intChannels[0] == intChannels[2]:
			self.netShortcut = None

		elif intChannels[0] != intChannels[2]:
			self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

	def forward(self, tenInput):
		if self.netShortcut is None:
			return self.netMain(tenInput) + tenInput

		elif self.netShortcut is not None:
			return self.netMain(tenInput) + self.netShortcut(tenInput)

class Downsample(torch.nn.Module):
	def __init__(self, intChannels):
		super().__init__()

		self.netMain = torch.nn.Sequential(
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.netMain(tenInput)
	# end
# end

class Upsample(torch.nn.Module):
	def __init__(self, intChannels):
		super().__init__()

		self.netMain = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.netMain(tenInput)
	# end
# end

class Refine(torch.nn.Module):
	def __init__(self, input_ch = 3):
		super().__init__()
		print("(Custom Network) Initialize Refine model")

		self.netImageOne = Basic('conv-relu-conv', [ input_ch, 24, 24 ])
		self.netImageTwo = Downsample([ 24, 48, 48 ])
		self.netImageThr = Downsample([ 48, 96, 96 ])

		self.netDisparityOne = Basic('conv-relu-conv', [ 1, 96, 96 ])
		self.netDisparityTwo = Upsample([ 192, 96, 96 ])
		self.netDisparityThr = Upsample([ 144, 48, 48 ])
		self.netDisparityFou = Basic('conv-relu-conv', [ 72, 24, 24 ])

		self.netRefine = Basic('conv-relu-conv', [ 24, 24, 1 ])
	# end

	def forward(self, tenImage, tenDisparity):
		tenMean = [ tenImage.view(tenImage.shape[0], -1).mean(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).mean(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]
		tenStd = [ tenImage.view(tenImage.shape[0], -1).std(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).std(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]

		tenImage = tenImage.clone()
		tenImage -= tenMean[0]
		tenImage /= tenStd[0] + 0.0000001

		tenDisparity = tenDisparity.clone()
		tenDisparity -= tenMean[1]
		tenDisparity /= tenStd[1] + 0.0000001

		tenImageOne = self.netImageOne(tenImage)
		tenImageTwo = self.netImageTwo(tenImageOne)
		tenImageThr = self.netImageThr(tenImageTwo)

		tenUpsample = self.netDisparityOne(tenDisparity)
		if tenUpsample.shape != tenImageThr.shape: tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageThr.shape[2], tenImageThr.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tenUpsample = self.netDisparityTwo(torch.cat([ tenImageThr, tenUpsample ], 1)); tenImageThr = None
		if tenUpsample.shape != tenImageTwo.shape: tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageTwo.shape[2], tenImageTwo.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tenUpsample = self.netDisparityThr(torch.cat([ tenImageTwo, tenUpsample ], 1)); tenImageTwo = None
		if tenUpsample.shape != tenImageOne.shape: tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageOne.shape[2], tenImageOne.shape[3]), mode='bilinear', align_corners=False) # not ideal
		tenUpsample = self.netDisparityFou(torch.cat([ tenImageOne, tenUpsample ], 1)); tenImageOne = None

		tenRefine = self.netRefine(tenUpsample)
		tenRefine *= tenStd[1] + 0.0000001
		tenRefine += tenMean[1]

		return torch.nn.functional.threshold(input=tenRefine, threshold=0.0, value=0.0)
