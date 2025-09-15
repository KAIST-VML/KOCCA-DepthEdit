"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19

import numpy as np


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class ContinuityLoss(nn.Module):
    def forward(self, seg, disp, threshold, weight = 1.0):
        ''' Compare discontinuity between segmap and the resulting depth '''
        delta_x = torch.abs((seg[:,:,:,1:] != seg[:,:,:,:-1]) * 1 \
            - (torch.abs(disp[:,:,:,1:] - disp[:,:,:,:-1]) > threshold) * 1)
        delta_y = torch.abs((seg[:,:,1:,:] != seg[:,:,:-1,:]) * 1 \
            - (torch.abs(disp[:,:,1:,:] - disp[:,:,:-1,:]) > threshold) * 1)

        total = torch.sum(delta_x) + torch.sum(delta_y)


        return weight * total / torch.sum(disp > -1)

class SemanticDepthSmoothness(nn.Module):
    ''' loss used in '''
    def forward(self, seg, disp):
        nc = seg.shape[1]
        loss_x, loss_y = torch.zeros_like(disp), torch.zeros_like(disp)
        for i in range(nc):
            seg_i = seg[:,i]
            loss_x += torch.abs(disp[:,:,:,1:] - disp[:,:,:,:-1]) * \
                                torch.abs(seg_i[:,:,:,1:] - seg_i[:,:,:,:-1])
            loss_y += torch.abs(disp[:,:,1:,:] - disp[:,:,:-1,:]) * \
                                torch.abs(seg_i[:,:,1:,:] - seg_i[:,::-1,:])
        total = torch.sum(loss_x) + torch.sum(loss_y)
        return total

class ReconstructionLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(ReconstructionLoss, self).__init__()
        self.weight = weight
    def forward(self, fake, real, mask):
        n = torch.sum(mask)
        return torch.sum(torch.abs(fake-real)*mask) / n

class MegaDepthLoss():
    def __init__(self,
                loss_lambda={'data':0.5, 'gradient':1.0},
                scales=[1,2,4,8],
                ):
        self.loss_lambda = loss_lambda
        self.loss_dict = {}
        self.scales = scales
        self.eps = 1e-6
        # self.is_inverse_depth = True
        # if data_type == 'megadepth':
        #     self.is_inverse_depth = False
        
    def __call__(self,output, target, mask=None, label_image = None):
        return self.total_loss(output, target, mask, label_image)

    def total_loss(self, output, target, mask=None, label_image = None):
        output_ = output
        target_ = target
        # if self.is_inverse_depth:
        #     output_ = 1 / (output_ + self.eps)
        #     target_ = 1 / (target_ + self.eps)
            
        log_output = torch.log(output_+self.eps)
        log_target = torch.log(target_+self.eps)
        R = log_output - log_target
        
        #mask
        if mask is None:
            mask = target < 1e-4
            mask = torch.add(torch.mul(mask,-1),1)
            R = R * mask
            self.N = torch.sum(mask)
            self.mask = mask
        else:
            mask = target < 1e-4
            mask = torch.add(torch.mul(mask,-1),1)
            self.N = torch.sum(mask)
            
            window_size = 7
            self.mask = get_weighted_mask(mask.squeeze(0), window_size)
        
        p_loss = self.data_loss(R) 
        g_loss = self.gradient_loss(R*self.mask)

        #if check local gradient for each label
        lg_loss = 0.0

        if not label_image is None:
            b,c,h,w = label_image.shape
            labels = set(label_image.cpu().detach().numpy().reshape(h*w,))
            for label in list(labels):
                label_mask = (label_image == label)
                lg_loss += self.gradient_loss(R * label_mask)
            self.loss_dict['local_gradient'] = lg_loss / len(labels)

        self.loss_dict['data'] = p_loss
        self.loss_dict['gradient'] = g_loss
        self.loss_dict['total'] = self.loss_lambda['data'] * p_loss + \
                                  self.loss_lambda['gradient'] * g_loss
        import numpy as np
        if np.isnan(self.loss_dict['total'].cpu().detach().numpy()):
            import pdb; pdb.set_trace()
        return self.loss_dict['total'], self.loss_dict

    def data_loss(self, R):
        # b,c,h,w = R.size()
        data = torch.sum(torch.pow(R, 2)) / self.N
        reg  = torch.pow(torch.sum(R), 2) / torch.pow(self.N, 2)
        return data - reg

    def gradient_loss(self, R):
        # b,c,h,w = R.size()
        # N = b*c*h*w
        total_grad = 0
        for scale in self.scales:
            # delta_x = torch.abs(R[:,:,:,:-1:scale] - R[:,:,:,1::scale])
            # delta_y = torch.abs(R[:,:,:-1:scale,:] - R[:,:,1::scale,:])
            delta_x = torch.abs(R[:,:,1::scale,:-1:scale] - R[:,:,1::scale,1::scale])
            delta_y = torch.abs(R[:,:,:-1:scale,1::scale] - R[:,:,1::scale,1::scale])
            # print(scale)
            # print('shape of x : {}'.format(delta_x.size()))
            # print('shape of y : {}'.format(delta_y.size()))
            total_grad += torch.sum(delta_x) + torch.sum(delta_y)
        
        return total_grad / self.N

# def dice_loss(pred, target, smooth = 1e-5):
#     # binary cross entropy loss
#     bce = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
    
#     pred = torch.sigmoid(pred)
#     intersection = (pred * target).sum(dim=(2,3))
#     union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
#     # dice coefficient
#     dice = 2.0 * (intersection + smooth) / (union + smooth)
    
#     # dice loss
#     dice_loss = 1.0 - dice
    
#     # total loss
#     loss = bce + dice_loss
    
#     return loss.sum(), dice.sum()

class MiDaSLoss(nn.Module):
    def __init__(self, loss_lambda={'data':0.5, 'gradient':1.0}, scales=[1,2,4,8]):
        super(MiDaSLoss, self).__init__()
        self.scales = scales
        self.loss_lambda = loss_lambda
        self.loss_dict = {}
        self.eps = 1e-6

    def ssitrim_loss(self, R):
        Um = int(0.8 * self.M)
        kth_value = torch.kthvalue(torch.flatten(R), Um).values

        return 1/(2*self.M) * torch.sum(R * (R < kth_value))

    def gradient_loss(self, R):
        total_grad = 0
        for scale in self.scales:
            delta_x = torch.abs(R[:,:,1::scale,:-1:scale] - R[:,:,1::scale,1::scale])
            delta_y = torch.abs(R[:,:,:-1:scale,1::scale] - R[:,:,1::scale,1::scale])
            total_grad += torch.sum(delta_x) + torch.sum(delta_y)
        
        return total_grad / self.M

    def scale_shift(self, d):
        t_d = torch.median(d)
        s_d = torch.sum(torch.abs(d - t_d)*self.valid_pixel) / self.M

        return (d - t_d) / s_d

    def forward(self, pred, target):
        self.valid_pixel = -1 * (target < 1e-4) + 1
        self.M = torch.sum(self.valid_pixel)

        ss_pred = self.scale_shift(pred)
        ss_target = self.scale_shift(target)

        R = torch.abs(ss_pred-ss_target)*self.valid_pixel

        data_loss = self.ssitrim_loss(R)
        grad_loss = self.gradient_loss(R)
        
        self.loss_dict['data'] = data_loss
        self.loss_dict['gradient'] = grad_loss
        self.loss_dict['total'] = self.loss_lambda['data'] * data_loss + \
                                  self.loss_lambda['gradient'] * grad_loss
       
        return self.loss_dict['total'], self.loss_dict

class DiceLoss(nn.Module):
    def forward(self, pred, target, mask = None, smooth = 1e-5):
        pred = pred.flatten()
        target = target.flatten()

        if mask is not None:
            mask = mask.flatten()
            pred = target*(mask == 0) + pred*(mask==1)
        
        intersection = (pred * target).sum()

        return 1 - ((2. * intersection + smooth) / ((pred + target).sum() + smooth))
    
# class ImageEdgeLoss(nn.Module):
#     def forward(self, image, pred, guide):
#         import kornia
#         magnitude, edge = kornia.filters.canny(image)
#         guide_mask = (guide != 0)

def get_weighted_mask(mask,window_size):
    assert len(mask.shape) == 3
    assert window_size % 2 == 1 # odd window size
    max_shift = window_size//2
    output = torch.zeros_like(mask)
    for i in range(-max_shift,max_shift+1):
        for j in range(-max_shift,max_shift+1):
            if i != 0 or j != 0:
                output += torch.roll(mask, (i,j), dims=(1,2))
    output = 1 - output/(window_size**2-1)
    return (output*mask).unsqueeze(0)


def EWMAE(work_image, ref_image, kappa=0.5):
    """GCMSE --- Gradient Conduction Mean Square Error.
    Computation of the GCMSE. An image quality assessment measurement 
    for image filtering, focused on edge preservation evaluation. 
    gcmse: float
        Value of the GCMSE metric between the 2 provided images. It gets
        smaller as the images are more similar.
    """
    # Normalization of the images to [0,1] values.
    max_val = ref_image.max()
    ref_image_float = ref_image.type(torch.cuda.FloatTensor)
    work_image_float = work_image.type(torch.cuda.FloatTensor)	
    normed_ref_image = ref_image_float / max_val
    normed_work_image = work_image_float / max_val
    
    # Initialization and calculation of south and east gradients arrays.
    gradient_S = torch.zeros_like(normed_ref_image)
    gradient_E = gradient_S.clone()
    gradient_S[:-1,: ] = torch.diff(normed_ref_image, axis=0)
    gradient_E[: ,:-1] = torch.diff(normed_ref_image, axis=1)
    
    # Image conduction is calculated using the Perona-Malik equations.
    cond_S = torch.exp(-(gradient_S/kappa) ** 2)
    cond_E = torch.exp(-(gradient_E/kappa) ** 2)
        
    # New conduction components are initialized to 1 in order to treat
    # image corners as homogeneous regions
    cond_N = torch.ones_like(normed_ref_image)
    cond_W = cond_N.clone()
    # South and East arrays values are moved one position in order to
    # obtain North and West values, respectively. 
    cond_N[1:, :] = cond_S[:-1, :]
    cond_W[:, 1:] = cond_E[:, :-1]
    
    # Conduction module is the mean of the 4 directional values.
    conduction = (cond_N + cond_S + cond_W + cond_E) / 4
    conduction = torch.clip (conduction, 0., 1.)
    G = 1 - conduction
    
    # Calculation of the GCMAE value 
    ewmae = (abs(G*(normed_ref_image - normed_work_image))).sum()/ G.sum()
    return ewmae

from distmap import euclidean_distance_transform as edt

def DepthBoundaryError(gt, pred, threshold=10):
    D = edt(1-gt) #euclidean distance transform matrix\
    D[D>threshold] = 0
    dbe_acc = (D*pred).sum()/(pred.sum()+1e-6)

    D_pred = edt(1-pred)
    # D_pred[D_pred>threshold] = 0
    dbe_comp = (D_pred*gt).sum()/(gt.sum()+1e-6)

    return dbe_acc, dbe_comp


def D3RError(gt, pred, threshold, indice_pool):
    def compute_ordinal_label2(ratios, threhold):
        ord_label = np.zeros_like(ratios)
        ord_label[ratios >= 1+threshold] = 1
        ord_label[ratios <= 1/(1+threshold)] = -1
        return ord_label
    
    gt = gt.flatten()
    pred = pred.flatten()

    valid = np.logical_and(gt[indice_pool[:,0]]!=0, gt[indice_pool[:,1]]!=0)
    indice_pool=indice_pool[valid]

    gt_ratios = gt[indice_pool[:,0]]/gt[indice_pool[:,1]]
    pred_ratios = pred[indice_pool[:,0]]/pred[indice_pool[:,1]]
    
    gt_ord = compute_ordinal_label2(gt_ratios, threshold) 
    pred_ord = compute_ordinal_label2(pred_ratios, threshold)

    
    ord_diff = (gt_ord != pred_ord) * 1
#     import pdb; pdb.set_trace()
    return ord_diff.sum() / len(ord_diff)
    
def OrdinalError(gt, pred, n, threshold, indice_pool=None):
    def compute_ordinal_label(ratio, threhold):
        if ratio >= 1+threshold:
            return 1
        elif ratio <= 1/(1+threshold):
            return -1
        else:
            return 0
    def compute_ordinal_label2(ratios, threhold):
        ord_label = np.zeros_like(ratios)
        ord_label[ratios >= 1+threshold] = 1
        ord_label[ratios <= 1/(1+threshold)] = -1
        return ord_label
    
    if indice_pool is None:
        indice = (gt != 0)
        indice_array = np.where(indice.flatten() != 0)[0]
        indice_pool = set()

        while len(indice_pool) < n:
            p0, p1= np.random.randint(0,len(indice_array),2)
            if p0 == p1: continue
            indice_pool.add((p0,p1))

        indice_pool = [*indice_pool]
    gt = gt.flatten()
    pred = pred.flatten()
#     for i in range(len(indice_array)):
#         for j in range(i+1, len(indice_array)):
#             indice_pool.append([indice_array[i],indice_array[j]])
    
#     gt_l = np.zeros((n))
#     pred_l = np.zeros((n))

    count = 0
#     for idx in indice_pool:
#         i,j = idx
#         gt_ratio = gt[i]/gt[j]
#         pred_ratio = pred[i]/pred[j]
#         gt_l[count] = compute_ordinal_label(gt_ratio,threshold)
#         pred_l[count] = compute_ordinal_label(pred_ratio,threshold)
#         count += 1
    gt_ratios = gt[indice_pool[:,0]]/gt[indice_pool[:,1]]
    pred_ratios = pred[indice_pool[:,0]]/pred[indice_pool[:,1]]
    
    gt_ord = compute_ordinal_label2(gt_ratios, threshold) 
    pred_ord = compute_ordinal_label2(pred_ratios, threshold)

    ord_diff = (gt_ord != pred_ord) * 1
#     import pdb; pdb.set_trace()
    return ord_diff.sum() / n
    
    
    