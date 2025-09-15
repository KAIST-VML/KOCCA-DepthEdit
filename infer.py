'''inference test for f-brs'''


import os, sys

import inference.optimizer as optimizer
from inference.predictor import get_predictor
import models.networks as networks
from options.train_options import TrainOptions
from options.test_options import TestOptions
import util.util as util

import torch
from torchvision import transforms
from torchvision.transforms import Compose

import PIL.Image as Image
import numpy as np
import random
import matplotlib.pyplot as plt

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

def normalize(input, scale=10):
    return (input-input.min())/(input.max()-input.min())*scale

def transform_image(image, input_size, interp_mode = Image.BICUBIC):
    transform = transforms.Compose([
    transforms.Resize(input_size, interpolation = interp_mode),
    transforms.ToTensor(),
    ])
    return transform(image)

def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()  

def create_segmap(label, opt, use_instance=False, isCustom=False):
    if isCustom:
        label[label>0] = label[label>0] -1 #if it is manually made
    label[label == 255] = opt.label_nc

    label_map = label.long()
    bs, _, h, w = label_map.size()
    nc = opt.label_nc + 1 if opt.contain_dontcare_label \
        else opt.label_nc

    input_label = torch.FloatTensor(1, nc, h, w).zero_()

    input_semantics = input_label.scatter_(1, label_map, 1.0)

    if use_instance:
        instance_edge_map = get_edges(label)
        return torch.cat([input_semantics, instance_edge_map], dim=1)
    else:
        return input_semantics

class Inference():
    def __init__(self, opt, seed=0):
        self.opt = opt
        set_seed(seed)
        self.dispScale = 10

    def load_model(self, weight, mode):
        self.mode = mode
        self.weight = weight

        if self.opt.experiment == "":
            if self.mode == "segDepth":
                seg_weight = '../semantic-segmentation-pytorch/ckpt/ade20k-hrnetv2-c1/decoder_epoch_30.pth'
                model = networks.SegDepthGenerator(self.opt, seg_weight)
            elif self.mode == "depth":
                model = networks.DepthGenerator(self.opt)  
            elif self.mode == 'segMap':
                seg_weight = '../semantic-segmentation-pytorch/ckpt/ade20k-hrnetv2-c1/decoder_epoch_30.pth'
                self.opt.decoder = "spade"
                model = networks.SegDepthGenerator(self.opt, seg_weight)
            elif self.mode == 'refine':
                self.opt.refine_depth = True
                model = networks.DepthGenerator(self.opt, 5)
        else:
            import models.networks.experiment.midas as midas
            if self.opt.experiment == 'MiDaS':
                model = midas.MidasGenerator(self.opt, pretrained=True)    
            elif self.opt.experiment == 'DPT-Large':
                model = midas.DPTGenerator(self.opt, pretrained=True)  

        self.model = model
        weights = torch.load(self.weight, torch.device('cpu'))
        self.model.load_state_dict(weights)
        self.model.eval()
    
    def infer(self, input_tensor, optimize=False):

        # '''Pre-process input'''
        # ratio = 256 / image.size[1]
        # input_size = (int(image.size[1] * ratio // 32 * 32),int(image.size[0] * ratio // 32 * 32))
        # image_tensor = transform_image(image, input_size, Image.NEAREST)[0:3,:].unsqueeze(0)
        # guide_tensor = transform_image(guide,input_size, Image.BICUBIC)[0].unsqueeze(0).unsqueeze(0) * self.dispScale
        # label_tensor = transform_image(label, input_size, Image.NEAREST)[0].unsqueeze(0).unsqueeze(0) * 255.0
        # seg = create_segmap(label_tensor, opt, use_instance=True, isCustom=isCustom)

        if optimize:
            ''' create predictor '''
            device = 'cpu'
            thresh = 0.49
            predictor_params = {'net_clicks_limit': 8}
            depth_predictor = get_predictor(self.model, device,
                            prob_thresh=thresh,
                            predictor_params=predictor_params)
            '''set location for scale and bias'''
            depth_predictor.location = self.opt.optimization_location

            guide_tensor = input_tensor[:,3,:,:].unsqueeze(0)
            input = torch.cat((input_tensor[:,:3,:,:], input_tensor[:,4:,:,:]),dim=1)

            with torch.no_grad():
                depth_predictor.set_input_image(input)

                pred = depth_predictor.get_prediction(guide_tensor)
                pred = normalize(pred)
                
                return pred
        else:
            with torch.no_grad():
                base = self.model(input_tensor)[0]
                base = base.detach().cpu().squeeze().numpy()
                base = normalize(base)
                import cv2
                cv2.imwrite("infer_result.png",(base/10*255).astype("uint8"))

                return base
        

def infer(opt, image, label, gt_depth, max_stroke=6, isCustom=False, seed=0):
    set_seed(seed)

    weight = '../semantic-segmentation-pytorch/ckpt/ade20k-hrnetv2-c1/decoder_epoch_30.pth'
    model = networks.SegDepthGenerator(opt,weight)
    # ckpt = './inference/27_net_G.pth'
    # ckpt = "checkpoints/segguideCoco_normalize_spade135BeforeCat-1.0_stroke_mobilenetEncoder_megadepth_10+4_30epoch_data1grad05_B16/latest_net_G.pth"
    ckpt = 'checkpoints/noSkipAtUp4n5_max10/latest_net_G.pth'
    '''spade'''
    # ckpt = "checkpoints/spade_stroke_mobilenetEncoder_megadepth_10+4_30epoch/30_net_G.pth"
    weights = torch.load(ckpt)
    model.load_state_dict(weights)
    model.eval()

    ratio = 256 / image.size[1]
    input_size = (int(image.size[1] * ratio // 32 * 32),int(image.size[0] * ratio // 32 * 32))
    image_tensor = transform_image(image, input_size, Image.NEAREST)[0:3,:].unsqueeze(0)
    depth_tensor = transform_image(gt_depth,input_size, Image.BICUBIC)[0].unsqueeze(0).unsqueeze(0)
    label_tensor = transform_image(label, input_size, Image.NEAREST)[0].unsqueeze(0).unsqueeze(0) * 255.0
    seg = create_segmap(label_tensor, opt, use_instance=True, isCustom=isCustom)
    depth_tensor = normalize(depth_tensor)
    disp = depth_tensor.squeeze().numpy()
    input = torch.cat((image_tensor,seg), dim=1)


    ''' create predictor '''
    device = 'cpu'
    thresh = 0.49
    predictor_params = {'net_clicks_limit': 8}
    depth_predictor = get_predictor(model, device,
                    prob_thresh=thresh,
                    predictor_params=predictor_params)

    with torch.no_grad():
        depth_predictor.set_input_image(input)

        guide = np.ones_like(disp)*-1
        model_input = torch.cat((image_tensor, torch.tensor(guide).unsqueeze(0).unsqueeze(0), seg),dim=1)
        base = model(model_input)[0]
        base = base.detach().squeeze().numpy()
        base = normalize(base)
        Image.fromarray(base/10*255).convert("L").save("base.png")
        import pdb; pdb.set_trace()

        for i in range(max_stroke):
            print("generate stroke ", i)
            length = random.randint(40, 50)
            thickness = random.randint(1,2)
            val = None
            while val is None:
                print('finding stroke')
                stroke_mask, val = util.sample_stroke(disp, guide, length, thickness, 0.1)
                print(val)
            guide[stroke_mask != 0] = val
            pred = depth_predictor.get_prediction(torch.tensor(guide).unsqueeze(0).unsqueeze(0))
            no_opt_pred = model(torch.cat((image_tensor, torch.tensor(guide).unsqueeze(0).unsqueeze(0), seg),dim=1))[0]
            pred = normalize(pred)
            no_opt_pred = normalize(no_opt_pred.detach().squeeze().numpy())

            initials = np.concatenate((disp,disp,base), axis=1)
            predictions = np.concatenate((no_opt_pred,guide,pred), axis=1)
            visualization = np.concatenate((initials,predictions), axis = 0)
            # plt.imsave("result_{}.png".format(str(i)),visualization)
            im = Image.fromarray(visualization/10*255) #already scaled by 10, so multiply 25 to make 250
            im.convert("L").save("result_{}.png".format(str(i)))


# if __name__ == '__main__':
#     opt = TrainOptions().parse()
#     # filename = '000000258554'
#     # filename = '000000409890'
#     # filename = '000000581789'
#     # folder = '/data1/coco-stuff/'; isCustom=False
#     filename='test'; folder='datasets/single/'; isCustom=True
#     depth = np.load(folder + 'train_img/' + filename + ".npy"); depth = Image.fromarray(depth)   
#     image = Image.open(folder + 'train_rgb/' + filename + '.jpg').convert('RGB')
#     label = Image.open(folder + 'train_label/' + filename + '.png')

#     infer(opt, image, label, depth, max_stroke=20, isCustom=isCustom, seed=7)


