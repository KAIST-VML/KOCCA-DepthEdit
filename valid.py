from matplotlib.transforms import Transform
from models import depth_model
from models.depth_model import DepthModel
from models.networks.sync_batchnorm import DataParallelWithCallback
from util.iter_counter import IterationCounter
import torch, torchvision
from tqdm import tqdm
import numpy as np

class BadPixelMetric:
    def __init__(self, threshold=1.25, depth_cap=10):
        self.__threshold = threshold
        self.__depth_cap = depth_cap

    def compute_scale_and_shift(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def __call__(self, prediction, target, mask):
        # transform predicted disparity to aligned depth
        target_disparity = torch.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / target[mask == 1]

        scale, shift = self.compute_scale_and_shift(prediction, target_disparity, mask)
        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        disparity_cap = 1.0 / self.__depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediciton_depth = 1.0 / prediction_aligned
        # print(prediction.shape, target.shape, mask.shape, prediciton_depth.shape)
        # bad pixel
        err = torch.zeros_like(prediciton_depth, dtype=torch.float)
        err[mask == 1] = torch.max(
            prediciton_depth[mask == 1] / target[mask == 1],
            target[mask == 1] / prediciton_depth[mask == 1],
        )

        err[mask == 1] = (err[mask == 1] > self.__threshold).float()

        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return 100 * torch.mean(p)

class Valid():
    def __init__(self, opt, valid_data_loader=None):
        self.opt = opt
        self.valid_data_loader = valid_data_loader
    
    def normalize(self, img):
        return (img-img.min())/(img.max()-img.min())

    def valid(self, depth_model, mode):
        print("Validation on dataset {} start".format(self.opt.dataset_mode))

        self.depth_model = depth_model
        self.depth_model.eval()

        self.fltAbsrel = []
        self.fltLogten = []
        self.fltSqrel = []
        self.fltRmse = []
        self.fltSiRmse = []
        self.fltThr1 = []
        self.fltThr2 = []
        self.fltThr3 = []

        self.midas_loss = []

        self.init_metrics()
        
        metric = BadPixelMetric()

        for batch_idx, batch in tqdm(enumerate(self.valid_data_loader)):            
            if len(self.opt.gpu_ids) == 0: 
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cpu()
            # load data
            if self.opt.dataset_mode == 'coco':
                depth = batch['disp']
            else:
                depth = batch['depth']

            # forward
            with torch.no_grad():
                if mode == 'ours':
                    loss, targets = self.depth_model(batch)
                    target = targets[0]
                elif mode == 'midas':
                    target = self.depth_model(batch['rgb'].to("cuda")).unsqueeze(0)

                if self.opt.dataset_mode != 'coco':
                    target = self.normalize(target)+0.5 # to make the disp range between 0.5~1.5
                    target_ = 1.0/(target+1e-6)
                else:
                    target_ = target.clone()

            _, h,w = batch['orig_depth'].shape
            import torch.nn.functional as F
            target = F.interpolate(target,size=(h,w),mode="bilinear",align_corners=False,)
            target_ = F.interpolate(target_,size=(h,w),mode="bilinear",align_corners=False,)
            depth = F.interpolate(depth,size=(h,w),mode="bilinear",align_corners=False,)    
            batch['valid'] = F.interpolate(batch['valid'],size=(h,w),mode="bilinear",align_corners=False,)
            batch['depth'] = F.interpolate(batch['depth'],size=(h,w),mode="bilinear",align_corners=False,)
            
            mask = batch['valid'][0,0,:,:].cpu().numpy()

            if not self.opt.dataset_mode == 'coco':
                loss = metric(target.cpu().squeeze(1), batch["depth"].cpu().squeeze(1), batch["valid"].cpu().squeeze(1))
                self.midas_loss.append(loss)

            target_ = target_.reshape((1,1,target_.shape[-2], target_.shape[-1]))
            npy_depth_ = target_[0, 0, :, :].cpu().numpy()
            
            # target_.dropna(inplace=True)
            npyLstsqa = np.stack([npy_depth_.flatten(), \
                np.full(npy_depth_.flatten().shape, 1.0, np.float32)]\
                , 1)
            npyLstsqb = depth.flatten()
            
            npyScalebias = np.linalg.lstsq(npyLstsqa, npyLstsqb, None)[0]
            npy_depth_ = (npy_depth_ * npyScalebias[0]) + npyScalebias[1]
            npy_depth = depth[0, 0, :, :].numpy()

            import matplotlib.pyplot as plt
            r,c = 2,3
            if batch_idx > -1:
                fig = plt.figure(figsize=(15,10))
                fig.add_subplot(r,c,4)
                plt.imshow(batch['label'].detach().cpu().numpy().squeeze())
                fig.add_subplot(r,c,2)
                plt.imshow(depth.detach().cpu().numpy().squeeze(),cmap='inferno')
                fig.add_subplot(r,c,3)
                plt.imshow((target_).detach().cpu().numpy().squeeze(),cmap='inferno')
                # plt.imshow(npy_depth_)
                fig.add_subplot(r,c,1)
                # plt.imshow(batch['rgb'].detach().cpu().numpy().squeeze().transpose((1,2,0)))
                plt.imshow(batch['orig_rgb'].detach().cpu().numpy().squeeze()/255.0)
                if 'guide' in batch and self.opt.max_stroke > 0:
                    fig.add_subplot(r,c,5)
                    plt.imshow(np.stack((batch['guide'].detach().cpu().numpy().squeeze(),)*1,axis=2),cmap='inferno')
                # plt.colorbar()
                # plt.show()
                plt.savefig("tta_result/tc2_{}".format(batch_idx))

            # For saving output if needed
            if batch_idx < 10:
                target_np = (target_).detach().cpu().squeeze().numpy()
                target_np = self.normalize(target_np)
                stacked = np.vstack((batch['orig_rgb'][0]/255.0,np.stack((target_np,)*3,axis=-1)))
                plt.imsave('tta_result/result_{}.png'.format(batch_idx), np.stack((target_np,)*3,axis=-1))

            # calc metrics
            self.calc_metrics(npy_depth, npy_depth_, mask)
            # self.calc_metrics(npy_depth, target_[0,0,:,:].cpu().numpy(), mask)

        print('abs_rel = ', sum(self.fltAbsrel) / len(self.fltAbsrel))
        # print('log10   = ', sum(self.fltLogten) / len(self.fltLogten))
        print('sq_rel  = ', sum(self.fltSqrel) / len(self.fltSqrel))
        print('rmse    = ', sum(self.fltRmse) / len(self.fltRmse))
        # print('si-RMSE = ', sum(self.fltSiRmse) / len(self.fltSiRmse))
        print('thr1    = ', sum(self.fltThr1) / len(self.fltThr1))
        print('thr2    = ', sum(self.fltThr2) / len(self.fltThr2))
        print('thr3    = ', sum(self.fltThr3) / len(self.fltThr3))
        print('midas   = ', sum(self.midas_loss) / len(self.midas_loss))

        self.valid_metrics['abs_rel'] = sum(self.fltAbsrel) / len(self.fltAbsrel)
        # self.valid_metrics['log10']   = sum(self.fltLogten) / len(self.fltLogten)
        self.valid_metrics['sq_rel']  = sum(self.fltSqrel) / len(self.fltSqrel)
        self.valid_metrics['rmse']     = sum(self.fltRmse) / len(self.fltRmse)
        # self.valid_metrics['si_rmse'] = sum(self.fltSiRmse) / len(self.fltSiRmse)
        self.valid_metrics['thr1'] = sum(self.fltThr1) / len(self.fltThr1)
        self.valid_metrics['thr2'] = sum(self.fltThr2) / len(self.fltThr2)
        self.valid_metrics['thr3'] = sum(self.fltThr3) / len(self.fltThr3)
        self.valid_metrics['midas'] = (sum(self.midas_loss) / len(self.midas_loss)).numpy()
        

    def init_metrics(self):
            self.valid_metrics = {}
            self.valid_metrics['abs_rel'] = 0
            self.valid_metrics['log10']   = 0
            self.valid_metrics['sq_rel']  = 0
            self.valid_metrics['rmse']     = 0
            self.valid_metrics['si_rmse'] = 0
            self.valid_metrics['thr1'] = 0
            self.valid_metrics['thr2'] = 0
            self.valid_metrics['thr3'] = 0
            self.valid_metrics['midas'] = 0

    def calc_metrics(self, gt, pred, mask):
        # print(gt)
        # print(pred)
        
        EPS = 1e-6
        valid_pixel_num = np.sum(mask)
        invalid_pixel_num = np.sum(1-mask)
        
        #print("valid/invalid pixel numb : {}/{}".format(valid_pixel_num, invalid_pixel_num))

        self.fltAbsrel.append((((pred - gt).__abs__() / (gt+EPS)).sum()/ valid_pixel_num).item())
        self.fltLogten.append( ( ((np.log10((pred+EPS)) - np.log10((gt+EPS))).__abs__()*mask).sum() / valid_pixel_num).item())
        self.fltSqrel.append( (((pred - gt).__pow__(2.0) / (gt+EPS) * mask).sum() / valid_pixel_num).item())
        self.fltRmse.append((np.sqrt((pred - gt).__pow__(2.0)).sum()/valid_pixel_num).item())
        self.fltThr1.append((np.maximum((pred / gt), (gt / pred)) < 1.25 ** 1).mean().item())
        self.fltThr2.append((np.maximum((pred / gt), (gt / pred)) < 1.25 ** 2).mean().item())
        self.fltThr3.append((np.maximum((pred / gt), (gt / pred)) < 1.25 ** 3).mean().item())
        
        R = (np.log(pred+EPS) - np.log(gt+EPS))* mask
        self.fltSiRmse.append(np.sqrt(np.sum(np.power(R, 2)) / valid_pixel_num - np.power(np.sum(R), 2) / np.power(valid_pixel_num, 2)))



    

# import argparse

# parser = argparse.ArgumentParser("")
# parser.add_argument("--mode", type=str, default='ours')
# parser.add_argument('')

# if __name__ == "__main__":
