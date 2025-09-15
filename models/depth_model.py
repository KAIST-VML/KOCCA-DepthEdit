import torch
import random
import models.networks as networks
import util.util as util

import kornia
import kornia.geometry.transform as gtf
import models.networks.midas_loss as midas_loss


class DepthModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        
        if self.opt.experiment == "":
            if self.opt.multi_head:
                self.model = networks.MultiHeadDepthGenerator(opt)
            elif self.opt.netG == 'segDepth':
                weight = '../semantic-segmentation-pytorch/ckpt/ade20k-hrnetv2-c1/decoder_epoch_30.pth'
                self.model = networks.SegDepthGenerator(opt, weight)
            elif self.opt.netG == "spade":
                self.model = networks.SPADEGenerator(opt)
            else:
                self.model = networks.DepthGenerator(opt)
        else:
            import models.networks.experiment.midas as midas
            if self.opt.experiment == 'MiDaS':
                self.model = midas.MidasGenerator(opt, pretrained=True)
            elif self.opt.experiment == 'DPT-Large':
                self.model = midas.DPTGenerator(opt, pretrained=True)
        
        if opt.isTrain:
            self.criterionMegadepth = networks.MegaDepthLoss()
            # self.criterionMiDaS = networks.MiDaSLoss()
            self.criterionMiDaS = midas_loss.ScaleAndShiftInvariantLoss()
            self.criterionContinuity = networks.ContinuityLoss()
            self.criterionNLL = torch.nn.NLLLoss(ignore_index=-1) #if two headed
            self.criterionCEL = torch.nn.CrossEntropyLoss()
            self.criterionMAE = torch.nn.L1Loss()
        
        if opt.isTrain and opt.continue_train:
            weights = torch.load(self.opt.ckpt)
            self.model.load_state_dict(weights)
            print("(Depth Model)    Done loading weight: {}".format(self.opt.ckpt))
        elif not opt.isTrain:
            weights = torch.load(self.opt.ckpt, torch.device('cpu'))
            self.model.load_state_dict(weights)
            print("(Depth Model)    Done loading weight: {}".format(self.opt.ckpt))

    def forward(self, data):
        input, real_image = self.preprocess_input(data)
        # if self.opt.multi_head:
        #     real_label = data['label']
        # else:
        #     real_label = None
        '''testing local megadepth'''
        if self.opt.segmap_type != "":
            real_label = data['label']
        else:
            real_label = None
        
        guide = data['guide'] if self.opt.l1_loss else None
        loss, generated = self.compute_generator_loss(input, real_image, real_label, guide)
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
            if self.opt.mask_rgb_with_guide:
                masked_rgb = data['rgb'] * torch.cat((((data['guide']==0)*1.0,)*3),dim=1)
                model_input = torch.cat([masked_rgb, data['guide']], dim=1)
            else:
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

                '''220523'''
                '''blur seg'''
                # input_semantics = kornia.filters.gaussian_blur2d(input_semantics,(9, 9), (5, 5))
                '''trimap by dilation'''
                # input_semantics_dilate = kornia.morphology.dilation(input_semantics, torch.ones((5,5)).cuda())
                # input_semantics = (input_semantics_dilate-input_semantics)*0.5 + input_semantics
                
                input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

            ''' other tps '''
            # if not random.randint(0,4):
            #     from util.util import apply_tps
            #     import itertools

            #     target_control_points = torch.Tensor(list(itertools.product(
            #         torch.arange(-1.0, 1.00001, 2.0 / 4),
            #         torch.arange(-1.0, 1.00001, 2.0 / 4),
            #     )))
            #     source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-0.1, 0.1)
                
            #     model_input = apply_tps(model_input, source_control_points, target_control_points, use_gpu=True)
            #     input_semantics = apply_tps(input_semantics, source_control_points, target_control_points, fill=self.opt.label_nc, use_gpu=True)
            #     data['disp'] = apply_tps(data['disp'], source_control_points, target_control_points, use_gpu=True)
            # '''-----------'''
            

            model_input = torch.cat([model_input, input_semantics], dim=1)

        '''kornia tps'''
        # def add_corners(t):
        #     corners = torch.Tensor([[[0.0,0.0],
        #                             # [0.0, 0.5],
        #                             [0.0,1.0],
        #                             # [0.5,0.0],[0.5,1.0],
        #                             [1.0,0.0],
        #                             # [1.0,0.5],
        #                             [1.0,1.0]]]).cuda()
            
        #     return torch.concat((t,corners), dim=1)
        # move_size = 0.1
        # points_src = (torch.rand(1, 3, 2)*0.5 + 0.3).cuda()
        # points_move = (torch.rand(1, 3, 2)*move_size - move_size/2).cuda()
        # points_dst = points_src+points_move
        # points_src = add_corners(points_src); points_dst = add_corners(points_dst)
        # kernel_weights, affine_weights = gtf.get_tps_transform(points_src, points_dst)
        # model_input = gtf.warp_image_tps(model_input, points_src, kernel_weights, 
		# 						affine_weights,align_corners=False)
        # data['disp']  = gtf.warp_image_tps(data['disp'], points_src, kernel_weights, 
		# 						affine_weights,align_corners=False)
        
        
        return model_input, data['disp']

    def compute_generator_loss(self, input, real_image, real_label=None,guide=None):
        G_losses = {}
        fake_image, fake_label = self.model(input)
        # print(fake_image)
        # print(fake_image.shape)
        # print(real_image)
        # print(real_image.shape)
        if not self.opt.isTrain:
            return G_losses, [fake_image,fake_label]

        if self.opt.megadepth_loss:
            G_losses['Megadepth'], loss_dict = self.criterionMegadepth(fake_image, real_image)
            
            G_losses['Gradient'] = loss_dict['gradient']
            G_losses['Data'] = loss_dict['data']
            G_losses['Total'] = torch.clone(G_losses['Megadepth'])

        if self.opt.midas_loss:
            G_losses['MiDaS'], loss_dict = self.criterionMiDaS(fake_image, real_image,flag='global')
            
            G_losses['Gradient'] = loss_dict['gradient']
            G_losses['Data'] = loss_dict['data']
            G_losses['Total'] = torch.clone(G_losses['MiDaS'])  
        
        if self.opt.l1_loss:
            guide_m = (guide!=0)*1
            if torch.count_nonzero(guide_m) != 0:
                G_losses['MAE'] = self.criterionMAE(fake_image*guide_m,real_image*guide_m)
                G_losses['Total'] += torch.clone(G_losses['MAE'])
                

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

        '''test 220401 local loss'''
        if self.opt.local_loss:
            if self.opt.megadepth_loss:
                local_key = 'Megadepth_local'
            else:
                local_key = 'MiDaS_local'
            G_losses[local_key] = 0
            count = 0
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            for i in range(nc):
                if not i in real_label:
                    continue
                local_target = real_image * (real_label==i)
                mask = local_target < 1e-4
                mask = torch.add(torch.mul(mask,-1),1)
                N = torch.sum(mask)
                if N == 0:
                    continue
                
                if self.opt.megadepth_loss:
                    local_loss , _= self.criterionMegadepth(fake_image*(real_label == i), real_image * (real_label==i))
                else:
                    local_loss , _= self.criterionMiDaS(fake_image*(real_label == i), real_image * (real_label==i))
                
                G_losses[local_key] += local_loss
                count += 1
            G_losses[local_key] /= count
            G_losses['Total'] += G_losses[local_key]

        return G_losses, [fake_image,fake_label]
    
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
    
    