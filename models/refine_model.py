import torch
import random
import models.networks as networks
import util.util as util
import models.networks.custom_network as custom_network


class RefineModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        
        if self.opt.experiment == "":
            if self.opt.multi_head:
                self.depth_model = networks.MultiHeadDepthGenerator(opt)
            elif self.opt.netG == 'segDepth':
                weight = '../semantic-segmentation-pytorch/ckpt/ade20k-hrnetv2-c1/decoder_epoch_30.pth'
                self.depth_model = networks.SegDepthGenerator(opt, weight)
            else:
                self.depth_model = networks.DepthGenerator(opt)
        else:
            if self.opt.experiment == 'MiDaS':
                import models.networks.experiment.midas as midas
                self.depth_model = midas.MidasGenerator(opt, pretrained=True)
        
        if opt.isTrain:
            self.criterionMegadepth = networks.MegaDepthLoss()
            self.criterionMiDaS = networks.MiDaSLoss(loss_lambda={'data':1.0, 'gradient':0.5})
            self.criterionContinuity = networks.ContinuityLoss()
            self.criterionNLL = torch.nn.NLLLoss(ignore_index=-1) #if two headed
            self.criterionCEL = torch.nn.CrossEntropyLoss()

            weights = torch.load(self.opt.ckpt, torch.device('cuda'))
            self.depth_model.load_state_dict(weights)

        # self.refine_model = custom_network.refineNet2(opt)
        self.model = custom_network.Refine()

        if not opt.isTrain:
            weights = torch.load(self.opt.ckpt, torch.device('cpu'))
            self.model.load_state_dict(weights)
            print("(Depth Model)    Done loading weight: {}".format(self.opt.ckpt))

    def forward(self, data, generated_depth = None):
        input, real_image = self.preprocess_input(data)
        real_label = data['label']
        loss, generated = self.compute_generator_loss(input, real_image, real_label)
        return loss, generated

    def preprocess_input(self, data):
        # move to GPU
        if self.use_gpu():
            data['disp'] = data['disp'].cuda()
            data['valid'] = data['valid'].cuda()
            data['rgb'] = data['rgb'].cuda()
            
        if self.opt.no_guide:
            model_input = data['rgb']
        else:
            if self.use_gpu():
                data['guide'] = data['guide'].cuda()
            model_input = torch.cat([data['rgb'], data['guide']], dim=1)

        if self.opt.refine_depth:
            '''
            In case initial depth is given
            modify GT depth to make initial depth
            '''
            data['flawed'] = data['flawed'].cuda()
            model_input = torch.cat((model_input, data['flawed']), dim=1)
            

        if self.opt.force_edge or self.opt.segmap_type != '':
            # data['instance'] = data['instance'].cuda()
            # inst_map = self.get_edges(data['instance'])
            # model_input = torch.cat([model_input, inst_map], dim=1)
            if self.use_gpu():
                data['label'] = data['label'].long().cuda()
            else:
                data['label'] = data['label'].long()
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()

            input_semantics = input_label.scatter_(1, label_map, 1.0)
            # '''220323 refine sky'''
            # sky_label = input_semantics[:,156,:,:]
            # sky_exist = sky_label.view(bs,-1).max(dim=1,keepdim=True)[0]
            # new_sky_label = sky_exist.repeat(1,h*w).reshape(bs,h,w)
            # input_semantics[:,156,:,:] = new_sky_label

            # concatenate instance map if it exists
            if not self.opt.no_instance:
                if self.use_gpu():
                    inst_map = data['instance'].cuda()
                else:
                    inst_map = data['instance']
                instance_edge_map = self.get_edges(inst_map)
                input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

            model_input = torch.cat([model_input, input_semantics], dim=1)

        return model_input, data['disp']

    def compute_generator_loss(self, input, real_image, real_label=None):
        G_losses = {}
        with torch.no_grad():
            gen_image, fake_label = self.depth_model(input)
            gen_image = gen_image.detach()
            gen_image.requires_grad_()

        fake_image = self.model(input[:,:3], gen_image)
        if not self.opt.isTrain:
            return G_losses, [fake_image,fake_label]

        if self.opt.megadepth_loss:
            G_losses['Megadepth'], loss_dict = self.criterionMegadepth(fake_image, real_image)
            
            G_losses['Gradient'] = loss_dict['gradient']
            G_losses['Data'] = loss_dict['data']
            G_losses['Total'] = torch.clone(G_losses['Megadepth'])

        if self.opt.midas_loss:
            G_losses['MiDaS'], loss_dict = self.criterionMiDaS(fake_image, real_image)
            
            G_losses['Gradient'] = loss_dict['gradient']
            G_losses['Data'] = loss_dict['data']
            G_losses['Total'] = torch.clone(G_losses['MiDaS'])  

        if self.opt.force_edge:
            # G_losses['Continuity'] = self.criterionContinuity(input[:,-1].unsqueeze(1), fake_image, 0.02)
            # G_losses['Continuity'] = self.criterionContinuity(fake_label, fake_image, 0.02)
            G_losses['Continuity'] = self.criterionContinuity(fake_label, fake_image, 0.008)
            G_losses['Total'] += G_losses['Continuity']

        if self.opt.multi_head:
            # G_losses['NLL'] = self.criterionNLL(fake_label,real_label.long().squeeze(1))
            # G_losses['Total'] += 0.5 * G_losses['NLL']
            debug = False
            G_losses['Cross Entropy'] = self.criterionCEL(fake_label, real_label.long().squeeze(1))
            G_losses['Total'] += 0.005 * G_losses['Cross Entropy']

        return G_losses, [gen_image,fake_image]
    
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