import os.path
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_mask
import torch, glob, cv2
import numpy as np
import h5py
import torchvision.transforms as transforms
import torch.utils.data as data
import PIL.Image as Image
import imageio

import random
import util.util as util

class KITTIDataset(BaseDataset):
    def __init__(self, opt, root='/data2/kitti_dataset/', is_train=True, transform = None, has_label=False, use_raw_depth = False):
        self.num_stroke = opt.max_stroke
        self.opt = opt
        self.transform = transform
        self.has_label = has_label
        self.use_raw_depth = use_raw_depth

        self.list_path = os.path.join(root, "eigen_test_files_with_gt.txt")
        
        with open(self.list_path, 'r') as f:
            filenames = f.readlines()
        
        self.data_rgb = []
        self.data_depth = []
        self.data_label = []
        for line in filenames:
            names = line.split()
            if names[1] != "None":
                image_name = os.path.join(root, names[0])
                if use_raw_depth:
                    depth_name = os.path.join(root,'data_depth_annotated/',names[1])
                else:
                    depth_name = os.path.join(root,'data_depth_filled/',names[1])

                if os.path.exists(image_name) and os.path.exists(depth_name):
                    self.data_rgb.append(image_name)
                    self.data_depth.append(depth_name)
                    if has_label:
                        self.data_label.append(depth_name.replace("/kitti_dataset/","/kitti_dataset/label_inferred_sseg/"))

        print("(KITTIDataset)   Total of {} files".format(len(self.data_rgb)))
        
    def __len__(self):
        return len(self.data_rgb)

    def __getitem__(self, index):
        def normalize(img):
            max_ = img.max()
            min_ = img.min()
            return (img-min_)/(max_-min_)
            
        # image
        rgb = np.array(imageio.imread(self.data_rgb[index], pilmode="RGB"))
        rgb = rgb; orig_rgb = rgb.copy()

        # depth and mask
        depth_png = np.array(imageio.imread(self.data_depth[index]), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255

        if self.use_raw_depth:
            depth = depth_png.astype(np.float) / 256.0;  
        else:
            depth = depth_png.astype(np.float) / 256.0 / 256.0 * 80.0
        orig_depth = depth.copy()

        # resize data
        target_size = 256
        if self.opt.experiment in ["MiDaS", "DPT-Large"] or self.opt.encoder == "MiDaS":
            target_size = 384
        # target_size = 1024
        ratio = target_size / min(depth.shape[1],depth.shape[0])
        self.input_size = (int(depth.shape[1] * ratio // 32 * 32),int(depth.shape[0] * ratio // 32 * 32))
        # self.input_size = (target_size, target_size)

        rgb = cv2.resize(rgb, self.input_size)
        depth = cv2.resize(depth, self.input_size)

        # depth = depth  / np.min(depth[depth>0])
        mask = depth != 0
        disp = np.zeros_like(depth)
        disp[mask==1] = 1.0 / depth[mask==1]
        
        
        if np.isnan(disp).any():
            exit("ERROR: disp has nan")
        if not self.transform is None:
            rgb_ = self.transform(rgb).squeeze().float()
        else:
            rgb_ = torch.from_numpy(rgb).float().permute(2,0,1)
        
        depth_ = torch.from_numpy(depth).float().unsqueeze(0)
        
        if self.has_label:
            # label_ = cv2.resize(np.array(Image.open(filename.replace("jpg","png"))), self.input_size)
            label_ = cv2.resize(np.array(Image.open(self.data_label[index])), self.input_size, interpolation=cv2.INTER_NEAREST)
            label_ = util.convert_kitti_label(label_)
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
            'filename': self.data_depth[index],
            }

        # after 0912
        data['disp'] = normalize(disp_)*10
        if self.num_stroke == -1:
            data['guide'] = data['disp']
        else:
            # disp = normalize(data['depth'].squeeze().numpy())
            disp = data['depth'].clone().squeeze().numpy()
            disp = disp/disp.max()*80
            if self.use_raw_depth:
                disp[mask==1] = 1/disp[mask==1]
            else:
                disp = 1/(disp+5)
            data['disp'] = torch.tensor(disp).clone().unsqueeze(0).float()
            disp = normalize(disp) * 10
            data['disp_'] = disp
            n = self.num_stroke
            if not self.use_raw_depth:
                # guide_layer = util.generate_point_guide(disp, n, self.opt.guide_empty, mask = mask)
                guide_layer, guide_pt = util.generate_stroke_guide(disp, n, self.opt.guide_empty) #sample 5 points           
            
            else:
                guide_layer = util.generate_point_guide(disp, n, self.opt.guide_empty, mask=mask) #sample 5 points           
            data['guide'] = torch.tensor(guide_layer).unsqueeze(0).float()

        return data

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(preprocess_mode='resize')
        parser.set_defaults(load_size=384)
        return parser