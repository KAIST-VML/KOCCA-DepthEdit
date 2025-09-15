import os.path
from data.image_folder import make_dataset
import torch, glob, cv2
import numpy as np
import h5py
import torchvision.transforms as transforms
import torch.utils.data as data
import PIL.Image as Image

import random
import util.util as util

class NYUDataset(data.Dataset):
    def __init__(self, opt, root='./NYUv2', is_train=True, transform = None, has_label=False):
        self.num_stroke = opt.max_stroke
        self.opt = opt
        self.transform = transform
        self.has_label = has_label
        
        if is_train:
            subdir = 'train'
        else:
            subdir = 'val'

        # set train/val root
        print(subdir)
        print(root)
        self.root = os.path.join(root, subdir)
        print('root dir : ', self.root)
        self.data = sorted(glob.glob(os.path.join(self.root,'*','*.h5')))
        print('data length : ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        def normalize(img):
            max_ = img.max()
            min_ = img.min()
            return (img-min_)/(max_-min_)
            
        objdata = h5py.File(self.data[index], 'r')
        depth = np.array(objdata['depth'], np.float32); orig_depth = depth.copy()
        rgb = np.array(objdata['rgb'], np.float32); orig_rgb = np.transpose(rgb.copy(),(1,2,0))

        filename = self.data[index].replace('h5','jpg')
        
        # resize data
        target_size = 256
        if self.opt.experiment in ["MiDaS", "DPT-Large"] or self.opt.encoder == "MiDaS":
            target_size = 384
        ratio = target_size / min(depth.shape[1],depth.shape[0])
        self.input_size = (int(depth.shape[1] * ratio // 32 * 32),int(depth.shape[0] * ratio // 32 * 32))
        # self.input_size = (target_size, target_size)

        rgb = cv2.resize(np.transpose(rgb, (1,2,0)), self.input_size)
        depth = cv2.resize(depth, self.input_size)

        # get invalid region
        mask = (depth > 0) & (depth < 10)
        # depth = depth  / np.min(depth[depth>0])
        disp = np.zeros_like(depth)
        disp[mask==1] = 1.0 / depth[mask==1]
        
        
        if np.isnan(disp).any():
            exit("ERROR: disp has nan")
        if not self.transform is None:
            rgb_ = self.transform(rgb).squeeze()
        else:
            rgb_ = torch.from_numpy(rgb).float().permute(2,0,1)
        
        depth_ = torch.from_numpy(depth).float().unsqueeze(0)
        
        if self.has_label:
            # label_ = cv2.resize(np.array(Image.open(filename.replace("jpg","png"))), self.input_size)
            label_ = cv2.resize(np.array(Image.open(filename.replace("official","class13").replace("jpg","png"))), self.input_size,
                                interpolation =cv2.INTER_NEAREST)
            from util.util import convert_nyu13_label
            label_ = convert_nyu13_label(label_)
            label_ = transforms.ToTensor()(label_)
            label_ = label_[0].unsqueeze(0) * 255
            label_[label_==255] = self.opt.label_nc
            instance_ = torch.clone(label_)
        else:
            label_ = torch.ones_like(depth_) * self.opt.label_nc
            instance_ = torch.ones_like(depth_) * self.opt.label_nc
        disp_ = torch.from_numpy(disp).float().unsqueeze(0)
        mask_ = torch.from_numpy(mask).float().unsqueeze(0)
        data = {'rgb': rgb_,
            'depth': depth_,
            'disp': disp_,
            'valid': mask_,
            'label': label_,
            'instance': instance_,
            'orig_depth': orig_depth,
            'orig_rgb': orig_rgb,
            'filename':filename,
            }

        # after 0912
        data['disp'] = normalize(disp_)*10
        if self.num_stroke == -1:
            data['guide'] = data['disp']
        else:
            disp = data['disp'].squeeze().numpy()
            disp = normalize(disp) * 10

            n = self.num_stroke
            guide_layer, guide_pt = util.generate_stroke_guide(disp, n, self.opt.guide_empty) #sample 5 points           
            data['guide'] = torch.tensor(guide_layer).unsqueeze(0)


        # before  0912
        # disp = data['disp'].squeeze().numpy()
        # disp = normalize(disp) * 10

        # n = self.num_stroke
        # guide_layer, guide_pt = util.generate_stroke_guide(disp, n) #sample 5 points           
        # data['guide'] = torch.tensor(guide_layer).unsqueeze(0)

        return data