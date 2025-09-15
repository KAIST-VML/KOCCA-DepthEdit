"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from cv2 import transform
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_mask
from PIL import Image
import util.util as util
import os
import numpy as np
import cv2, torch, random, math
import torchvision.transforms as transforms
import kornia

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths, rgb_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        util.natural_sort(rgb_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)
        
        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]
        rgb_paths = rgb_paths[:opt.max_dataset_size]

        # if not opt.no_pairing_check:
        #     for path1, path2 in zip(label_paths, image_paths):
        #         assert self.paths_match(path1, path2), \
        #             "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths
        self.rgb_paths = rgb_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        rgb_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths, rgb_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=transforms.InterpolationMode.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

#         # input image (real images)
#         image_path = self.image_paths[index]
#         assert self.paths_match(label_path, image_path), \
#             "The label_path %s and image_path %s don't match." % \
#             (label_path, image_path)
#         image = Image.open(image_path)
#         image = image.convert('RGB')
        
        # input image (depth images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        
        if ".npy" in image_path:
            image = np.load(image_path)
#             image = (image-image.min())/(image.max()-image.min()) * 255
            image = Image.fromarray(image)
        else:
            image = Image.open(image_path)

        orig_depth = torch.Tensor(np.array(image))

        transform_image = get_transform(self.opt, params, method=transforms.InterpolationMode.BICUBIC, normalize=False)
        image_tensor = transform_image(image)
        '''normalize'''
        disp_rescale = 10
        image_tensor = (image_tensor - image_tensor.min())/(image_tensor.max()-image_tensor.min()) * disp_rescale

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)
                
         # original rgb image
        rgb_path = self.rgb_paths[index]

        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, rgb_path)

        rgb = Image.open(rgb_path)
        rgb = rgb.convert('RGB'); orig_rgb = rgb.copy()

        # transform_image = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        if self.opt.encoder in ["MiDaS","ResNet101"] or self.opt.experiment in ["MiDaS", "DPT-Large"]:
            transform_image = get_transform(self.opt, params, method=transforms.InterpolationMode.NEAREST, normalize=True)
        else:
            transform_image = get_transform(self.opt, params, method=transforms.InterpolationMode.NEAREST, normalize=False)
        rgb_tensor = transform_image(rgb)
        '''edge'''
        if self.opt.decoder == "spade" and self.opt.encoder == "MobileNetV2": #testing for spade
            # rgb_path = label_path
            gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_BGR2GRAY)
            # gray_blur = cv2.GaussianBlur(gray,(3,3), sigmaX=0, sigmaY=0)
            edge = cv2.Canny((gray).astype(np.uint8), 100, 200)
            edge_tensor = transform_image(Image.fromarray(edge))
            rgb_tensor = torch.cat((edge_tensor,)*3, dim=0)

        if self.opt.train_mode == 'seg2depth':
            input_dict = {'label': label_tensor,
                        'instance': instance_tensor,
                        'image': image_tensor,
                        'rgb': rgb_tensor,
                        # 'path': image_path,
                        }
        else:
            mask = image_tensor < 1e-4
            mask = 1 - (1*mask)
            input_dict = {
                        'disp': image_tensor,
                        'rgb': rgb_tensor,
                        # 'path': image_path,
                        'valid': mask,
                        }
            if not self.opt.isTrain:
                orig_rgb = get_transform(self.opt, params, method=transforms.InterpolationMode.NEAREST, normalize=False)(orig_rgb)
                input_dict['orig_rgb'] = orig_rgb.permute((1,2,0))*255
                input_dict['orig_depth'] = (image_tensor).squeeze(0)
            # randomly sample depth points
            if not self.opt.no_guide:
                # '''point as guide'''
                # disp = input_dict['disp']
                # size = self.opt.crop_size #if coco, it will be 256
                # indices = self.random_sample(disp, size, 5) #sample 5 points

                # guide_layer = torch.zeros_like(disp)
                # for h,w in indices:
                #     guide_layer[:,h,w] = disp[:,h,w]
                #     input_dict['guide'] = guide_layer # B C H W
                
                '''stroke as guide'''
                disp = input_dict['disp'].squeeze().numpy()
                if self.opt.isTrain and self.opt.max_stroke > 0:
                    n = random.randint(0,self.opt.max_stroke)
                else:
                    n = self.opt.max_stroke
                if self.opt.refine_depth: n = 1
                
                if n >= 0:
                    guide_layer, guide_pt = util.generate_stroke_guide(disp, n, self.opt.guide_empty) #sample 5 points           
                    input_dict['guide'] = torch.tensor(guide_layer).unsqueeze(0)
                else:
                    guide = input_dict['disp'].clone()
                    blur_size = random.randint(8,12)
                    guide = kornia.filters.box_blur(guide.unsqueeze(0),(blur_size,blur_size))
                    input_dict['guide'] = guide.squeeze(0)

            if self.opt.refine_depth:
                guide = input_dict['guide'].unsqueeze(0)
                mask = torch.zeros_like(guide)
                mask[guide>-1] = 1
                mask[mask!=1] = 0
                # mask = kornia.filters.gaussian_blur2d(mask, (401,401), (80,80))
                radius = random.randrange(20,60)
                mask = self.gaussian_blob(mask, guide_pt, radius, 0, sigma=0.6).reshape(mask.shape)
                incomplete_disp = input_dict['disp']+(mask*random.sample([-1,1],1)[0]*5)
                input_dict['flawed'] = incomplete_disp.squeeze(0)

            if self.opt.force_edge or self.opt.segmap_type != "" or self.opt.multi_head:
                use_label = True
                if self.opt.random_label:
                    use_label = random.randint(0,1)
                if use_label:
                    input_dict['instance'] = instance_tensor
                    input_dict['label'] = label_tensor
                else:
                    input_dict['label'] = torch.ones_like(label_tensor)*self.opt.label_nc
                    input_dict['instance'] = (torch.ones_like(label_tensor)*self.opt.label_nc).long()

                
        #mask if exist
        if self.opt.mask != '':
            orig_mask = Image.open(self.opt.mask)
            mask = get_transform_mask(self.opt, image, orig_mask)

            dilated_mask = cv2.dilate(mask.squeeze().numpy(),np.ones((3,3),np.uint8),iterations=5)
            dilated_mask = torch.from_numpy(dilated_mask).unsqueeze(0)

            input_dict['mask'] = mask
            input_dict['dilated_mask'] = dilated_mask

        # Give subclasses a chance to modify the final output
        if 'label' in input_dict:
            self.postprocess(input_dict)
        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def random_sample(self, image, size, n):
        sample_indices = []

        for i in range(n):
            index  = torch.tensor(random.sample(range(size),2))
            sample_indices.append(index)

        return sample_indices

    def gaussian_blob(self, image, pt, radius, size, sigma=0.4):
        if len(image.shape) == 4:
            b,c,h,w = image.shape
        else: h,w = image.shape

        pt = torch.Tensor(pt)
        num_pt = len(pt)
        yMat = pt[:,0]
        xMat = pt[:,1]

        rowMat = torch.arange(h).reshape((h,1,1)).expand((h,w,num_pt))
        colMat = torch.arange(w).reshape((1,w,1)).expand((h,w,num_pt))
        distMat = ((rowMat-yMat)**2 + (colMat-xMat)**2)**0.5

        outMat = 1/(sigma*math.sqrt(2*math.pi)) * \
            math.e ** (-1 * ((distMat-size)/(radius-size))**2 / (2*sigma**2))
        outMat[distMat<size] = 1
        out = torch.amax(outMat,axis=2) 
        return out
    
