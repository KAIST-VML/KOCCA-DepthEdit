'''
dataloader for 3d kenburn synthetic dataset
'''
import glob, os

import json
import math
# import sys
import cv2
import torch
import numpy as np
import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset

import kornia
import random


class KenburnDataset(Pix2pixDataset):
    def get_paths(self, opt, root='/data2/kenburns/', is_train=True):
        self.opt = opt
        # self.input_size = (384, 384)
        self.input_size = (256, 256)
        self.is_train = is_train
        self.rgb_dir = os.path.join(root, 'rgb')
        self.depth_dir = os.path.join(root, 'depth')

        # rgb and depth folders
        rgb_folders = os.listdir(self.rgb_dir)
        depth_folders = os.listdir(self.depth_dir)
        
        # error raise when folder number does not match
        assert len(rgb_folders) == len(depth_folders), \
            'the number of folder does not match ( rgb: {}, depth:{})'\
                .format(len(rgb_folders),len(depth_folders))

        # rgb and depth file dirs
        print('\n (dataloader) ===> data dir of rgb and depth: \n\trgb - {}\n\tdetph - {}'\
            .format(self.rgb_dir, self.depth_dir))

        self.rgbs = sorted(glob.glob(os.path.join(self.rgb_dir, '*', '*.png')))
        self.depths = sorted(glob.glob(os.path.join(self.depth_dir, '*', '*.exr')))
        
        # error raise when folder number does not match
        # assert len(self.rgbs) == len(self.depths), \
        #     'the number of files does not match ( rgb: {}, depth:{})'\
        #         .format(len(self.rgbs),len(self.depths))

        # # check img data
        # from tqdm import tqdm
        # for i in tqdm(range(len(self.rgbs))):
        #     rgb_filename = os.path.split(self.rgbs[i])[-1]
        #     depth_filename = os.path.split(self.depths[i])[-1]

        #     # print(i)
        #     # print("rgb : {}\ndepth: {}".format(rgb_filename, depth_filename))
            
        #     rgb_filename = rgb_filename.split('-image.png')[0]
        #     depth_filename = depth_filename.split('-depth.exr')[0]
            
        #     # print("rgb : {}\ndepth: {}".format(rgb_filename, depth_filename))
        #     assert depth_filename == rgb_filename, \
        #         'the file name does not match : {}, {}'.format(depth_filename, rgb_filename)

        # if self.is_train and not self.configs.no_augment:
        #     self.augment = AugmentationPipeline()
        print('\n (dataloader) ===> finished initializing the 3DKenburns dataset ...\n\n')
        return self.rgbs, self.depths, self.depths, self.rgbs

    def __len__(self):
        return len(self.rgbs)
        
    def __getitem__(self, index):
        img = self.load_img(self.rgbs[index])
        fov = self.load_meta(self.rgbs[index])
        depth = self.load_depth(self.depths[index])
        
        # resize data
        img = cv2.resize(img, self.input_size)
        depth = cv2.resize(depth, self.input_size)

        # normalize depth and get disp
        # depth = depth  / np.min(depth[depth>0])
        depth, disp = self.depth2disp(depth, fov)
        # disp = 1 / depth
        # depth = depth / np.max(depth)
        
        # heuristics
        # when depth is lower than 0.01 consider as sky mask
        mask = disp < 1e-4
        mask = 1 - mask 
        
        # to torch
        img_ = torch.from_numpy(img).float().permute(2,0,1)
        disp_ = torch.from_numpy(disp).float().unsqueeze(0)
        depth_ = torch.from_numpy(depth).float().unsqueeze(0)
        mask_ = torch.from_numpy(mask).float().unsqueeze(0)

        # augment
        # if self.is_train and not self.configs.no_augment:
            # img_ = self.augment(img_)[0, ...]
        img_ = torch.clamp(img_, min=0.0, max=1.0)
        
        data = {'rgb': img_,
            'depth': depth_,
            'disp': disp_,
            'valid': mask_}

        disp = data['disp'].squeeze().numpy()
        if self.opt.isTrain and self.opt.max_stroke > 0:
            n = random.randint(0,self.opt.max_stroke)
        else:
            n = self.opt.max_stroke
        if self.opt.refine_depth: n = 1
        
        if n >= 0:
            guide_layer, guide_pt = util.generate_stroke_guide(disp, n, self.opt.guide_empty) #sample 5 points           
            data['guide'] = torch.tensor(guide_layer).unsqueeze(0)
        else:
            guide = data['disp'].clone()
            blur_size = random.randint(8,12)
            guide = kornia.filters.box_blur(guide.unsqueeze(0),(blur_size,blur_size))
            data['guide'] = guide.squeeze(0)

        data['label'] = data['guide'].clone()
        data['instance'] = data['guide'].clone()
        return data

    def load_meta(self, png_dir_):
        dirs_ = os.path.split(png_dir_)[:-1]
        png_filename = os.path.split(png_dir_)[-1]
        png_filename_ = png_filename.split('-image.png')[0]
        meta_filename = str(png_filename_[:-3] + '-meta.json')
        root = str(dirs_[0])
        fov = json.loads(open(os.path.join(root, meta_filename), 'r').read())['fltFov']
        # print(fov)
        return fov

    def load_depth(self, dir_):
        depth = cv2.imread(filename=dir_, flags=-1)[:, :, None]
        depth = np.ascontiguousarray(depth.astype(np.float32))
        # tmpdepth = 
        return depth
        # self.depth2disp(depth, fov)
        
    def load_img(self, dir_):
        img = cv2.imread(filename=dir_, flags=-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        npyImages = np.ascontiguousarray(img.astype(np.float32))
        return npyImages * (1.0 / 255.0)

    @staticmethod
    def depth2disp(depth, Fov):
        baseline = 40.0 # fixed value for 3d kenburns data
        focal_length = 0.5 * 512 * math.tan(math.radians(90.0) - (0.5 * math.radians(Fov)))
        new_depth = depth / (focal_length * baseline)
        new_depth = new_depth  / np.min(new_depth[new_depth>0])
        return new_depth, 1 / new_depth

if __name__ == "__main__":
    root = '/mnt/hdd_4t/data/kenburns/'
    from torch.utils.data import DataLoader

    ### create dataset
    dataset =  DepthDataloader(root)
    batch_size = 4
    threads = 0

    train_epoch_size = int(len(dataset) / int(batch_size))
    data_loader = DataLoader(dataset=dataset, num_workers= threads, \
        shuffle=False, batch_size=batch_size, pin_memory=True, drop_last=False)

    data_iter = iter(data_loader)