import torch
import random
import models.networks as networks
import util.util as util
import models.networks.custom_network as custom_network
import kornia


class DeblurModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        
        self.model = custom_network.Refine(input_ch=1) #input = 1 channel edge
        
        if opt.isTrain:
            self.criterionMegadepth = networks.MegaDepthLoss()
        
        if opt.isTrain and opt.continue_train:
            weights = torch.load(self.opt.ckpt)
            self.model.load_state_dict(weights)
            print("(Depth Model)    Done loading weight: {}".format(self.opt.ckpt))
        elif not opt.isTrain:
            weights = torch.load(self.opt.ckpt, torch.device('cpu'))
            self.model.load_state_dict(weights)
            print("(Depth Model)    Done loading weight: {}".format(self.opt.ckpt))

    def forward(self, data, generated_depth = None):
        input, real_image = self.preprocess_input(data)
        loss, generated = self.compute_generator_loss(input, real_image)
        return loss, generated

    def preprocess_input(self, data):
        # move to GPU
        if self.use_gpu():
            data['disp'] = data['disp'].cuda()
            data['valid'] = data['valid'].cuda()
            data['rgb'] = data['rgb'].cuda()
            
        if self.opt.no_guide:
            model_input = data['disp']
        else:
            if self.use_gpu():
                data['guide'] = data['guide'].cuda()
            model_input = torch.cat([data['disp'], data['guide']], dim=1)

        if self.opt.refine_depth:
            blurred_disp = kornia.filters.gaussian_blur2d(data['disp'],(21,21),(1.5,1.5))
            magnitude, edge = kornia.filters.canny(data['rgb'])
            data['flawed'] = blurred_disp
            model_input = torch.cat([edge,blurred_disp], dim=1)

        return model_input, data['disp']

    def compute_generator_loss(self, input, real_image, real_label=None):
        G_losses = {}
        fake_image = self.model(input[:,0].unsqueeze(1), input[:,1].unsqueeze(1))
        if not self.opt.isTrain:
            return G_losses, [fake_image,fake_label]

        if self.opt.megadepth_loss:
            G_losses['Megadepth'], loss_dict = self.criterionMegadepth(fake_image, real_image)
            
            G_losses['Gradient'] = loss_dict['gradient']
            G_losses['Data'] = loss_dict['data']
            G_losses['Total'] = torch.clone(G_losses['Megadepth'])


        return G_losses, [fake_image,input[:,1].unsqueeze(1)]
    
    def create_optimizers(self, opt):
        params = list(self.model.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer = torch.optim.Adam(params, lr=G_lr, betas=(beta1, beta2))
        return optimizer

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def save(self, epoch):
            util.save_network(self.model, 'G', epoch, self.opt)


    #util
    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()