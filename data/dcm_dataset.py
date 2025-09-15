import os.path
from data.image_folder import make_dataset
import torch, glob, cv2
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import PIL.Image as Image
import kornia

import random
import util.util as util


class DCMDataset(data.Dataset):
    def __init__(self, opt, root='/Users/jungeunyoo/Documents/DCM/ComicsDepth/data/dcm_cropped/', 
                is_train=True, transform = None, has_label=False,
                num_pt=20, kernel_size=7, shift=0, pseudo_path = None):

        self.root = root
        self.num_stroke = opt.max_stroke
        self.opt = opt
        self.transform = transform
        self.has_label = has_label
        self.is_train = is_train
        self.num_pt = num_pt
        self.kernel_size = kernel_size
        self.shift = shift
        self.pseudo_path = pseudo_path
        self.data_pseudo = []

        if is_train:
            self.data = sorted(glob.glob("/data1/DCM/ComicsDepth/data/dcm_cropped/images/**/*.jpg"))
        else:
            self.data = sorted(glob.glob("/data1/DCM/ComicsDepth/data/dcm_cropped/depth/**/*.txt"))
            if pseudo_path is not None:
                self.data_pseudo = [os.path.join(pseudo_path,"{}_depth.png".format(i)) for i in range(len(self.data))]
                print("Load {} pseudo gt from path: {}".format(len(self.data_pseudo),pseudo_path))
            if has_label:
                count = 0
                for data_i in self.data:
                    if os.path.exists(data_i.replace(".txt",".png").replace("depth","label")): count+=1
                print("Found {} label".format(count))
            print('data length : ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        def normalize(img):
            max_ = img.max()
            min_ = img.min()
            return (img-min_)/(max_-min_)
            
        depth_ = self.data[index]
        rgb = depth_.replace("/depth/","/images/").replace(".txt",".jpg")
        rgb = cv2.imread(rgb)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        ''' if pseudo gt is given'''
        if len(self.data_pseudo) == len(self.data):
            disp = cv2.imread(self.data_pseudo[index])/255.0
            disp = disp[:,:,0]
            disp = normalize(disp) * 10
        '''in certain ratio, inference does not work'''
        # orig_ratio= rgb.shape[0] / rgb.shape[1]
        # if orig_ratio > 2.0 or orig_ratio < 0.5: return {'rgb':'hi'}
        ''' remove above if only used for testing 117 images'''

        # resize data
        target_size = 256
        if self.opt.experiment == "MiDaS" or self.opt.encoder == "MiDaS":
            target_size = 384
        ratio = target_size / min(rgb.shape[1],rgb.shape[0])
        self.input_size = (int(rgb.shape[1] * ratio // 32 * 32),int(rgb.shape[0] * ratio // 32 * 32))
        # self.input_size = (target_size, target_size)
        
        rgb_ = cv2.resize(rgb, self.input_size)

        if not self.transform is None:
            rgb_ = self.transform(rgb_).squeeze()
        else:
            rgb_ = torch.from_numpy(rgb).float().permute(2,0,1)
        #add sharpness
        import kornia
        # rgb_ = kornia.enhance.sharpness(rgb_, 2)s
        
        label_path = depth_.replace("/depth/","/label/").replace(".txt",".png")
        if self.has_label and os.path.exists(label_path):
            label_ = cv2.resize(np.array(Image.open(label_path)), self.input_size,
                                interpolation =cv2.INTER_NEAREST)
            label_ = transforms.ToTensor()(label_)
            label_ = label_[0].unsqueeze(0) * 255
            label_[label_==255] = self.opt.label_nc
            instance_ = torch.clone(label_)
        else:
            label_ = torch.ones_like(rgb_)[0].unsqueeze(0) * self.opt.label_nc
            instance_ = torch.ones_like(label_) * self.opt.label_nc
        # magnitude, edge = kornia.filters.canny(rgb_.unsqueeze(0))
        # instance_ = edge.squeeze(0)

        data = {'rgb': rgb_.float(),
            'depth': depth_,
            'label': label_.float(),
            'instance': instance_.float(),
            'disp': rgb_,
            'valid': rgb_,
            'orig_rgb': rgb
            }
        # data['guide'] = (torch.ones_like(label_) * -1.0).float()
        if self.num_pt == 0 and self.pseudo_path is None:
            guide = np.ones((rgb.shape[0],rgb.shape[1]))*self.opt.guide_empty
        else:
            if self.pseudo_path is None: 
                guide = self.sample_guide_points(depth_,(rgb.shape[0],rgb.shape[1]),
                            num_pt=self.num_pt,dilate_kernel_size=self.kernel_size)
            else:
                guide, guide_pt = util.generate_point_guide(disp, 20, self.opt.guide_empty)
        guide = transforms.ToTensor()(cv2.resize(guide, (rgb_.shape[2],rgb_.shape[1]), interpolation = cv2.INTER_NEAREST)).float()
        data['guide'] = guide

        return data
    

    @staticmethod
    def read_depth_ordering(filepath):
        with open(filepath) as file:
            lines = file.readlines()
        n = len(lines)
        points = []
        for i in range(n):
            line = lines[i].replace("\n","")
            l1, l2, x, y = line.split(" ")
            l1, l2, x, y = int(l1), int(l2), int(x), int(y)
            approx_depth = l1*10 + l2
            points.append((x,y,approx_depth))
            
        return points

    @staticmethod
    def convert_depth_to_disparity(depth_ord):
        approx_depth = depth_ord[:,2]
        max_, min_ = max(approx_depth), min(approx_depth)
        approx_disp = (approx_depth * -1) + max_+ min_
        disp_ord = depth_ord.copy()
        disp_ord[:,2] = approx_disp
        
        return disp_ord, max_, min_


    @staticmethod
    def generate_guide_map(points, size, shift):
        depth = np.zeros(size)
        for pt in points:
            x,y,d = pt
            if shift > 0:
                x, y = random.randint(x-shift,x+shift), random.randint(y-shift,y+shift) 
                x = min(size[1]-1, max(x,0))
                y = min(size[0]-1, max(y,0))
            depth[y][x] = d
        
        return depth

    def sample_guide_points(self, filepath, size, num_pt=-1, dilate_kernel_size = -1):
        points = self.read_depth_ordering(filepath)
        if len(points) == 0:
            return np.ones(size)*self.opt.guide_empty

        points, max_d, min_d = self.convert_depth_to_disparity(np.array(points))
        if num_pt >= 0:
            # ind = np.random.choice(list(range(len(points))),num_pt)
            ind = random.sample(range(len(points)),num_pt)
            sampled_points = np.array(points)[ind]
        elif num_pt == -2: # every other points
            sampled_points = [points[i] for i in range(len(points)) if i%2 == 0]
        else:
            sampled_points = points

        depth = self.generate_guide_map(sampled_points,size, self.shift)

        if dilate_kernel_size != -1:
            depth = cv2.dilate(depth, np.ones((dilate_kernel_size,)*2))
        
        mask = (depth == 0)*1
        disp_scale = 10
        depth = (depth - min_d)/(max_d-min_d) * disp_scale
        if self.opt.guide_empty < 0.0:
            depth[mask==1] = self.opt.guide_empty
        else:
            # depth[depth==0.0] = 1e-6
            depth[depth<0.1] = 0.1
            depth[mask==1] = 0.0

        return depth