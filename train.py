"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from options.test_options import TestOptions
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from trainers.pix2pix_trainer import Pix2PixTrainer
from trainers.depth_trainer import DepthTrainer

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

'''parse options'''
opt = TrainOptions().parse()

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

# '''set seed'''
# # set_seed(9876)
# set_seed(98765)
# # set_seed(987654)
# # set_seed(123)



# print options to help debugging
print(' '.join(sys.argv))

'''set seed for dataloader'''
set_seed(9876)
# load the dataset
if opt.valid_size > 0:
    dataloader, valid_dataloader = data.create_two_dataloader(opt, opt.valid_size)
else:
    dataloader = data.create_dataloader(opt)
    valid_dataloader = None
    # valid_opt = TestOptions().parse()
    # valid_dataloader = data.create_dataloader(valid_opt)

'''set seed for training'''
set_seed(opt.seed)
'''create trainer for our model'''
if opt.train_mode == 'seg2depth':
    trainer = Pix2PixTrainer(opt)
else:
    trainer = DepthTrainer(opt, dataloader, valid_dataloader) 

if opt.train_mode == 'depth':
    trainer.train()
    exit()

'''create tool for counting iterations'''
iter_counter = IterationCounter(opt, len(dataloader))

#save logs
import datetime
log_dir = 'logs/' + datetime.datetime.now().strftime("%y%m%d_") + opt.name
writer = SummaryWriter(log_dir=log_dir)

from torchvision.utils import make_grid
        
def normalize(t):
    return (t-torch.min(t))/(torch.max(t)-torch.min(t))


for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    print("({}) Epoch {}".format(opt.name,epoch))

    # Training
    trainer.pix2pix_model.train()
    for i, data_i in tqdm(enumerate(dataloader, start=iter_counter.epoch_iter)):
        if valid_dataloader is not None and \
            iter_counter.total_steps_so_far % 1000 == 0: #do validation
            print("(Train) Validation Start at step: ",iter_counter.total_steps_so_far)
            trainer.pix2pix_model.eval()
            
            d_loss = []
            g_loss = []
            total_loss=[]
            
            rgbs = None
            gens = None
            gts = None

            for i, val_data_i in tqdm(enumerate(valid_dataloader)):
                with torch.no_grad():
                    #trainer.run_generator_one_step(val_data_i)
                    g_losses, generated = trainer.pix2pix_model(val_data_i, mode='generator')
                    if opt.megadepth_loss:
                        d_loss.append(g_losses['Data'].cpu().numpy())
                        g_loss.append(g_losses['Gradient'].cpu().numpy())
                        total_loss.append(g_losses['Megadepth'].cpu().numpy())
                    
                    if rgbs is None:
                        rgbs = val_data_i['rgb']
                        gens = normalize(generated.cpu())
                        # gts = normalize(val_data_i['image'])
                    else:
                        rgbs = torch.cat([rgbs,val_data_i['rgb']],0)
                        gens = torch.cat([gens,normalize(generated.cpu())],0)
                        # gts = torch.cat([gts,normalize(val_data_i['image'])],0)

            rgbs = torch.cat([rgbs[:5,:,:,:],rgbs[5:,:,:,:]],2)
            gens = torch.cat([gens[:5,:,:,:],gens[5:,:,:,:]],2)
            writer.add_image('Valid/RGB', make_grid(rgbs), iter_counter.total_steps_so_far)
            writer.add_image('Valid/Generated', make_grid(gens), iter_counter.total_steps_so_far)             
            if opt.megadepth_loss:
                val_losses = {}
                val_losses['Gradient'] = sum(g_loss)/len(g_loss)
                val_losses['Data'] = sum(d_loss)/len(d_loss)
                val_losses['Total'] = sum(total_loss)/len(total_loss)

                print('Gradient Loss = ', val_losses['Gradient'])
                print('Data Loss = ', val_losses['Data'])
                print('Total Loss = ', val_losses['Total'])

                for key, value in val_losses.items():
                    writer.add_scalar('Valid/'+key, value,iter_counter.total_steps_so_far)
            
            print(" (Train) Validation Ended. Start Training")
        trainer.pix2pix_model.train()

        iter_counter.record_one_iteration()

        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)
        
        '''does not need to train discriminator for disparity generation'''
#         # train discriminator
#         trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.total_steps_so_far % 50 == 0:

            gen = normalize(trainer.generated).cpu()
            gen = torch.cat([gen,gen,gen],1)
            gt = normalize(data_i['image'])
            gt = torch.cat([gt,gt,gt],1)
            
            rgb = data_i['rgb']
            col1 = torch.cat([rgb,gt],2) 

            if opt.mask != '':

                ############################################
                #   RGB |   masked label  |     mask
                #   GT  |   generated     |     composited
                ############################################

                mask = data_i['mask']
                masked_label = data_i['label']*mask
                label = torch.cat([masked_label,masked_label,masked_label],1)
                col2 = torch.cat([label,gen],2)

                msk = torch.cat([mask,mask,mask],1)
                # masked_gen =  gt*(1-mask) + gen*mask # composited
                masked_gen = gt*(mask == 0)
                masked_gen += gen*(mask==1)
                col3 = torch.cat([msk,masked_gen],2)
                writer.add_image('Results', make_grid(torch.cat([col1,col2,col3])), iter_counter.total_steps_so_far)
            else:

                ######################
                #   RGB |   label
                #   GT  |   generated
                ######################

                label = torch.cat([data_i['label'],data_i['label'],data_i['label']],1)
                col2 = torch.cat([label,gen],2)
                writer.add_image('Results', make_grid(torch.cat([col1,col2])), iter_counter.total_steps_so_far)

        if iter_counter.total_steps_so_far % 10 == 0:
            for key, value in trainer.g_losses.items():
                writer.add_scalar('G_loss/'+key, value,iter_counter.total_steps_so_far)

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
