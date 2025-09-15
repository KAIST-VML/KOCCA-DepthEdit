"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from configparser import Interpolation
import re
import importlib
import torch
from argparse import Namespace
import numpy as np
from PIL import Image
import os
import argparse
import dill as pickle
import util.coco



def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# returns a configuration for creating a generator
# |default_opt| should be the opt of the current experiment
# |**kwargs|: if any configuration should be overriden, it can be specified here


def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        if N == 182:  # COCO
            important_colors = {
                'sea': (54, 62, 167),
                'sky-other': (95, 219, 255),
                'tree': (140, 104, 47),
                'clouds': (170, 170, 170),
                'grass': (29, 195, 49)
            }
            for i in range(N):
                name = util.coco.id2label(i)
                if name in important_colors:
                    color = important_colors[name]
                    cmap[i] = np.array(list(color))

    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


###############################################################################
# Code by Jey
###############################################################################

def random_sample(self, image, size, n):
    sample_indices = []

    for i in range(n):
        index  = torch.tensor(random.sample(range(size),2))
        sample_indices.append(index)

    return sample_indices

def generate_point_guide(depth,num_pt=-1,empty_value=-1.0,mask=None, thickness=None):
    if thickness is None:
        thickness = random.randint(7,9)
    # thickness = random.randint(20,25)
    def generate_guide_map(depth, points):
            guide = np.zeros_like(depth)
            for pt in points:
                y,x = pt
                guide[y,x] = depth[y,x]
            
            return guide
    
    if mask is None:
        # pool = list(zip(range(depth.shape[0]),range(0,depth.shape[0])))
        points=set()
        while len(points) < num_pt:
            y = random.randint(thickness//2,depth.shape[0]-thickness//2)
            x = random.randint(thickness//2,depth.shape[1]-thickness//2)
            points.add((y,x))
        points = list(points)
    else:
        y, x = (mask==1).nonzero()
        pool = list(zip(y,x))
    
        points = np.array(pool)[np.random.choice(range(len(pool)),num_pt)]

    guide = generate_guide_map(depth, points)
    if thickness != -1:
            guide = cv2.dilate(guide, np.ones((thickness,)*2))

    return guide, points

def sample_stroke(depth, guide, length, thickness, max_diff, empty_value = -1.0, mask=None):

    sampled = np.zeros_like(depth)
    empty_value = np.float32(empty_value)

    #sample a point

    # y = random.randint(length,depth.shape[0]-length-1)
    # x = random.randint(0,depth.shape[0]-length-1)
    # point = np.array([y,x])
    # val = depth[point[0],point[1]]
    
    if mask is None:
        y = random.randint(length,depth.shape[0]-length-1)
        x = random.randint(0,depth.shape[0]-length-1)
        point = np.array([y,x])
        # pool = list(zip(range(length,depth.shape[0]-length-1),range(0,depth.shape[0]-length-1)))
    else:
        y, x = (mask==1).nonzero()
        indices = ((y >= length)&(y<depth.shape[0]-length-1)&(x<depth.shape[0]-length-1)).nonzero()
        pool = list(zip(y[indices],x[indices]))
    
        point = np.array(random.sample(pool,1))[0]
    val = depth[point[0],point[1]]

    # cur_point = point.clone()
    cur_point = point
    direction_set = np.array((
                            ((1,0),(0,1)), #down right
                            ((-1,0),(0,1)), #up right
                            ))

    directions = direction_set[random.randint(0,1)]
    indices = [0,1]
    p1 = round(random.random(),1); p2 = 1- p1

    sampled_length = 0
    sampled_pt = []

    for i in range(length):
        index = np.random.choice(indices, p=[p1,p2])
        direction = directions[index]
        h, w = cur_point + direction
        if guide[h,w] == empty_value and abs(val - depth[h,w]) < max_diff:
            sampled[h,w] = 1
            cur_point += direction
            sampled_length += 1
            sampled_pt.append((h,w))
        else:
            direction = directions[1-index]
            h,w = cur_point + direction
            if guide[h,w] == empty_value and abs(val - depth[h,w]) < max_diff:
                sampled[h,w] = 1
                cur_point += direction
                sampled_length += 1
                sampled_pt.append((h,w))
            else:
                break
    #thicken
    kernel = np.ones((3,3), np.uint8)
    sampled = cv2.dilate(sampled, kernel, iterations=thickness)

    if sampled_length < length * 0.8:
        return sampled, None, sampled_pt
    # print("Original length: ", length)
    # print("generated stroke with length {} and thickness {}".format(sampled_length, thickness))
    return sampled, val, sampled_pt

import random, cv2
import PIL.Image as Image  

def generate_stroke_guide(depth, n, empty_value = -1.0, mask=None, thickness=None):
    '''
    Generate n strokes

    depth = depth/disparity map
    n = number of strokes
    empty_value = value for empty space (default -1.0)
    '''
    h,w = depth.shape[-2], depth.shape[-1]
    H,W = (384,384)
    if h > 256 or w > 256:
        depth = cv2.resize(depth, dsize=(H,W), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask*1, dsize=(H,W), interpolation=cv2.INTER_NEAREST)

    # Initially fill guide map with empty value
    guide = np.ones_like(depth) * empty_value 

    guide_pt = []
    while n > 0:
        length = random.randint(40, 50)
        if thickness is None:
            thickness = random.randint(1,2)
        stroke_mask, val, sampled_pt = sample_stroke(depth, guide, length, thickness, 0.1, empty_value, mask)
        if val != None:
            if empty_value == 0.0 and val == 0.0: val = 1e-6
            guide[stroke_mask != 0] = val
            guide_pt += sampled_pt
            n -= 1
    
    if h > 256 or w > 256:
        guide = cv2.resize(guide, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    
    return guide, guide_pt

def sample_priority_point(gt, pred, mask, dilate_kernel_size = -1):
        radius = 20 #int(np.ceil(dilate_kernel_size/2))+1
        depth = np.zeros_like(gt)
        
        diff = abs(gt-pred) * mask
        diff[:radius,:] = 0.0; diff[-radius:,:]=0.0; diff[:,:radius]=0.0; diff[:,-radius:]=0.0
        target_val = diff.max()
        target_idx = np.where(diff==target_val)
        if len(target_idx[0]) > 1:
            target_idx = (target_idx[0][0],target_idx[0][1])
        # sample gt guide from most errored region
        gt_val = gt[target_idx]
        depth[target_idx] = gt_val if gt_val > 0.1 else 0.1
        

        if dilate_kernel_size != -1:
            depth = cv2.dilate(depth, np.ones((dilate_kernel_size,)*2))

        return depth 

def perturb_stroke(guide, guide_empty=0.0):
    mask = (guide != guide_empty)
    min_, max_ = guide[mask].min(), guide[mask].max()
    guide *= guide # squared values
    guide = (guide - guide.min())/(guide.max()-guide.min())
    guide = (guide * (max_-min_))
    guide[mask] += min_
    
    return guide

def convert_171_to_182(index):
    CLASSES_171 = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
        'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
        'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood')
    
    label = CLASSES_171[index]
    CLASSES_DICT_182 = np.load("label.npy", allow_pickle=True).item()

    if label in CLASSES_DICT_182:
        return CLASSES_DICT_182[label]
    else:
        return 255 #unknown label

def get_label_dict(filename):
    import csv
    labelDict = {}
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            labelDict[int(row[1])] = int(row[2])
    return labelDict


def convert_label(label):
    unique_idx = np.unique(label)
    new_label = np.zeros_like(label)
    for idx in unique_idx:
        new_label[label==idx] = convert_171_to_182(idx)
    
    return new_label

def convert_nyu13_label(label):
    unique_idx = np.unique(label)
    label_dict={0:255,1:64,2:83,3:101,4:61,5:114,6:122,7:255,8:172,9:62,10:164,11:71,12:172,13:180}
    new_label = np.zeros_like(label)
    for idx in unique_idx:
        new_label[label==idx] = label_dict[idx]
    
    return new_label

def convert_kitti_label(label):
    unique_idx = np.unique(label)
    label_dict= get_label_dict("./util/misc/cityscape2coco.csv")
    new_label = np.zeros_like(label)
    for idx in unique_idx:
        new_label[label==idx] = label_dict[idx]
    
    return new_label
    

'''
https://github.com/WarBean/tps_stn_pytorch/blob/master/tps_grid_gen.py
'''
import torch
import itertools
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

class TPSGridGen(nn.Module):

    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate) # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim = 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate

def grid_sample(input, grid, canvas = None, use_gpu = False):
    if use_gpu:
        grid = grid.cuda()
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        if use_gpu:
            canvas = canvas.cuda()
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

def apply_tps(input, source_control_points, target_control_points, fill=None, use_gpu=False):
    batch_size, channel, image_height, image_width = input.shape
    tps = TPSGridGen(image_height, image_width, target_control_points)
    source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
    grid = source_coordinate.view(1, image_height, image_width, 2)
    grid = torch.cat((grid,)*batch_size, dim=0)
    if fill != None:
        canvas = Variable(torch.Tensor(batch_size, channel, image_height, image_width).fill_(fill))
        target_image = grid_sample(input, grid, canvas, use_gpu=use_gpu)
    else:
        target_image = grid_sample(input, grid, use_gpu=use_gpu)

    return target_image


"""
Source: https://github.com/isl-org/MiDaS
"""

def read_image(path):
    """Read image and output RGB image (0-1).
    Args:
        path (str): path to file
    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

def write_depth(path, depth, bits=1, absolute_depth=False):
    """Write depth map to pfm and png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    if absolute_depth:
        out = depth
    else:
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2**(8*bits))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return