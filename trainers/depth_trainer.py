from models import depth_model

from models.depth_model import DepthModel
from models.refine_model import RefineModel
from models.deblur_model import DeblurModel

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.networks import custom_network

from util.iter_counter import IterationCounter
import torch, torchvision
from tqdm import tqdm
import numpy as np

from torch import distributed as dist

def set_seed(seed):
    from torch.backends import cudnn
    import numpy as np
    import random
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class DepthTrainer():
    def __init__(self, opt, train_data_loader, valid_data_loader=None):
        self.opt = opt
        if self.opt.refine_depth: 
            # self.depth_model = RefineModel(opt)
            self.depth_model = DeblurModel(opt)
        else:
            self.depth_model = DepthModel(opt)
        
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        if len(opt.gpu_ids) > 1: # Multi GPU training
            from apex.parallel import DistributedDataParallel as DDP
            self.depth_model = DDP(self.depth_model.cuda())
            self.opt.batchSize *= len(opt.gpu_ids)
            self.depth_model_on_one_gpu = self.depth_model.module

        elif len(opt.gpu_ids)==1: # One GPU training
            self.depth_model = DataParallelWithCallback(self.depth_model,
                                                        device_ids = opt.gpu_ids)
            self.depth_model_on_one_gpu = self.depth_model.module
        else:
            self.depth_model_on_one_gpu = self.depth_model

        if opt.isTrain:
            self.optimizer = self.depth_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

            import datetime, inspect
            from torch.utils.tensorboard import SummaryWriter

            log_dir = opt.log_dir + datetime.datetime.now().strftime("%y%m%d_") + opt.name
            self.writer = SummaryWriter(log_dir=log_dir)

            '''wandb setting'''
            if not opt.debug:
                import wandb
                attributes = inspect.getmembers(opt, lambda a:not(inspect.isroutine(a)))
                optionDict = {}
                for a in attributes:
                    if not (a[0].startswith('__') and a[0].endswith('__')):
                        optionDict[a[0]] = a[1]
                wandb.init(project="Interactive Depth", name=opt.name, entity="jeyoo", config=optionDict)


    def train(self):
        self.iter_counter = IterationCounter(self.opt, len(self.train_data_loader))
        set_seed(self.opt.seed)
        for epoch in self.iter_counter.training_epochs():
            self.iter_counter.record_epoch_start(epoch)
            print("({}) Epoch {}".format(self.opt.name,epoch))
            
            #Training
            self.train_epoch()

            self.update_learning_rate(epoch)
            self.iter_counter.record_epoch_end()

            if epoch % self.opt.save_epoch_freq == 0 or epoch == self.iter_counter.total_epochs:
                if self.opt.single_data: continue
                print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, self.iter_counter.total_steps_so_far))
                self.save('latest')
                self.save(epoch)
                
    def train_epoch(self):
        for i, data_i in tqdm(enumerate(self.train_data_loader, start=self.iter_counter.epoch_iter)):
            # try validation for every 1000 steps
            if self.iter_counter.total_steps_so_far % 1000 == 0:
                print("(Train) Validation Start at step: ", self.iter_counter.total_steps_so_far)
                self.valid_epoch()

            self.depth_model.module.model.train() #train only the desired model
            
            self.iter_counter.record_one_iteration()
            self.run_one_step(data_i)
            if self.iter_counter.total_steps_so_far % 500 == 0:
                #visualize
                gen = self.generated[0]
                gt = data_i['disp']
                rgb = data_i['rgb']
                mask = data_i['valid']
                guide =  data_i['guide'] if 'guide' in data_i else None
                if self.opt.refine_depth:
                    import kornia
                    magnitude, edge = kornia.filters.canny(data_i['rgb'])
                    guide = edge
                if self.opt.mask_rgb_with_guide:
                    rgb = data_i['rgb'] * torch.cat((((data_i['guide']==0)*1.0,)*3),dim=1)
                # flawed = data_i['flawed'] if self.opt.refine_depth else None
                flawed = self.generated[1]
                self.write_imgs(rgb, gt, gen, mask, guide, flawed, None, self.iter_counter.total_steps_so_far)

            if self.iter_counter.total_steps_so_far % 10 == 0:
                for key, value in self.losses.items():
                    self.writer.add_scalar('G_loss/'+key, value,self.iter_counter.total_steps_so_far)
            
            if not self.opt.debug:
                wandb.log(self.losses)

    def valid_epoch(self):
        self.depth_model.eval()

        d_loss = []
        g_loss = []
        total_loss=[]

        if self.opt.valid_size > 0:
            data_loader = self.valid_data_loader
        else:
            data_loader = self.train_data_loader
        
        for i, val_data_i in tqdm(enumerate(data_loader)):
            with torch.no_grad():
                losses, generated = self.depth_model(val_data_i)
                d_loss.append(losses['Data'].cpu().numpy())
                g_loss.append(losses['Gradient'].cpu().numpy())
                total_loss.append(losses['Total'].cpu().numpy())

                rgb = val_data_i['rgb']
                gen = generated[0]
                gt = val_data_i['disp']
                mask = val_data_i['valid']
        
        guide =  val_data_i['guide'] if 'guide' in val_data_i else None
        gen_label = generated[1]
        if gen_label is not None:
            _, gen_label = torch.max(gen_label, dim = 1)
            gen_label = gen_label.unsqueeze(1)
        # if len(self.opt.gpu_ids) > 1 and dist.get_rank() == 0:
        self.write_imgs(rgb, gt, gen, mask, guide, None, gen_label, self.iter_counter.total_steps_so_far, mode='valid')
        
        valid_losses={}
        valid_losses['Data'] = sum(g_loss)/len(g_loss)
        valid_losses['Gradient'] = sum(g_loss)/len(g_loss)
        valid_losses['Total'] = sum(total_loss)/len(total_loss)

        for key, value in valid_losses.items():
                self.writer.add_scalar('Valid/'+key, value,self.iter_counter.total_steps_so_far)
    
    def valid(self):
        print("Validation on dataset {} start".format(self.opt.dataset_mode))
        self.depth_model.eval()

        self.fltAbsrel = []
        self.fltLogten = []
        self.fltSqrel = []
        self.fltRmse = []
        self.fltSiRmse = []
        self.init_metrics()

        from tqdm import tqdm
        for batch_idx, batch in tqdm(enumerate(self.valid_data_loader)):            
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cpu()
            # load data
            if self.opt.dataset_mode == 'coco':
                depth = batch['disp']
            else:
                depth = batch['depth']
            mask = batch['valid'][0,0,:,:].cpu().numpy()

            # forward
            with torch.no_grad():
                loss, targets = self.depth_model(batch)
                target_ = targets[0]
                if self.opt.dataset_mode != 'coco':
                    target_ = 1.0/target_
            
            npy_depth_ = target_[0, 0, :, :].cpu().numpy()
            #npy_depth_[np.logical_not(np.isnan(npy_depth_))] = 1e6
            npy_depth_[np.isnan(npy_depth_)] = 1e6

            # target_.dropna(inplace=True)
            npyLstsqa = np.stack([npy_depth_.flatten(), \
                np.full(npy_depth_.flatten().shape, 1.0, np.float32)]\
                , 1)
            npyLstsqb = depth.flatten()
            
            npyScalebias = np.linalg.lstsq(npyLstsqa, npyLstsqb, None)[0]
            npy_depth_ = (npy_depth_ * npyScalebias[0]) + npyScalebias[1]
            npy_depth = depth[0, 0, :, :].numpy()
            
            # calc metrics
            self.calc_metrics(npy_depth, npy_depth_, mask)

        print('abs_rel = ', sum(self.fltAbsrel) / len(self.fltAbsrel))
        print('log10   = ', sum(self.fltLogten) / len(self.fltLogten))
        print('sq_rel  = ', sum(self.fltSqrel) / len(self.fltSqrel))
        print('rmse    = ', sum(self.fltRmse) / len(self.fltRmse))
        print('si-RMSE = ', sum(self.fltSiRmse) / len(self.fltSiRmse))

        self.valid_metrics['abs_rel'] = sum(self.fltAbsrel) / len(self.fltAbsrel)
        # self.valid_metrics['log10']   = sum(self.fltLogten) / len(self.fltLogten)
        self.valid_metrics['sq_rel']  = sum(self.fltSqrel) / len(self.fltSqrel)
        self.valid_metrics['rms']     = sum(self.fltRmse) / len(self.fltRmse)
        self.valid_metrics['si_rmse'] = sum(self.fltSiRmse) / len(self.fltSiRmse)
        

    def init_metrics(self):
            self.valid_metrics = {}
            self.valid_metrics['abs_rel'] = 0
            self.valid_metrics['log10']   = 0
            self.valid_metrics['sq_rel']  = 0
            self.valid_metrics['rms']     = 0
            self.valid_metrics['si_rmse'] = 0

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
        self.fltRmse.append((np.sqrt((pred - gt).__pow__(2.0)).sum())/valid_pixel_num.item())

        pred[pred<1] = 1
        gt[gt<1] = 1
        R = (np.log(pred) - np.log(gt))* mask
        self.fltSiRmse.append(np.sqrt(np.sum(np.power(R, 2)) / valid_pixel_num - np.power(np.sum(R), 2) / np.power(valid_pixel_num, 2)))


    def run_one_step(self,data):
        self.optimizer.zero_grad()
        losses, generated = self.depth_model(data)
        total_loss = losses['Total']
        total_loss.backward()
        self.optimizer.step()
        self.losses = losses
        self.generated = generated

        # #refine
        # if self.iter_counter.total_steps_so_far > 120000:
        #     self.optimizer_refine.zero_grad()
        #     generated_depth = generated[0].detach()#.requires_grad_()
        #     # refine_input = torch.cat((generated_depth, data['rgb']), dim=1)
        #     # loss, depth_refined = self.depth_model(data, refine_input)
        #     loss, depth_refined = self.depth_model(data, generated_depth)
        #     # if self.iter_counter.total_steps_so_far % 500 == 0:
        #     #     import pdb; pdb.set_trace()
        #     loss.backward()
        #     self.losses['refine_loss'] = loss
        #     self.optimizer_refine.step()
        #     self.generated[0] = depth_refined


    def get_latest_generated(self):
        return self.generated

    # def update_learning_rate(self, epoch):
    #     self.update_learning_rate(epoch)

    def save(self, epoch):
        self.depth_model_on_one_gpu.save(epoch)

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            # for param_group in self.optimizer_D.param_groups:
            #     param_group['lr'] = new_lr_D
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def normalize(t):
        return (t-torch.min(t))/(torch.max(t)-torch.min(t))

    def write_imgs(self, rgb, target, output, mask, guide, flawed, label, iter_, mode = 'train'):
        
        #send resulted image to cpu
        output = output.detach().cpu()
        
        if mask is not None:
            target_ = target*mask
            output_ = output*mask

        target_grid = torchvision.utils.make_grid(target_, nrow=target.size(0)//2, normalize=True, scale_each=True)
        output_grid = torchvision.utils.make_grid(output_, nrow=output.size(0)//2, normalize=True, scale_each=True)
        rgb_grid = torchvision.utils.make_grid(rgb, nrow=rgb.size(0)//2)

        if mode == 'valid':
            self.writer.add_images('valid/output', output_grid, iter_, dataformats='CHW')
            self.writer.add_images('valid/input', rgb_grid, iter_, dataformats='CHW')
            if label is not None:
                label_grid = torchvision.utils.make_grid(label, label=rgb.size(0)//2)
                self.writer.add_images('valid/output_label', label_grid, iter_,  dataformats='CHW')
        else:
            self.writer.add_images('depth/gt', target_grid, iter_, dataformats='CHW')
            self.writer.add_images('depth/output', output_grid, iter_, dataformats='CHW')
            self.writer.add_images('rgb/input', rgb_grid, iter_, dataformats='CHW')

            if guide is not None:
                guide_grid = torchvision.utils.make_grid(guide, nrow=guide.size(0)//2, normalize=True, scale_each=True)
                self.writer.add_images('depth/guide', guide_grid, iter_, dataformats='CHW' )
            if flawed is not None:
                flawed_grid = torchvision.utils.make_grid(flawed, nrow=guide.size(0)//2, normalize=True, scale_each=True)
                self.writer.add_images('depth/gt_flawed', flawed_grid, iter_, dataformats='CHW' )