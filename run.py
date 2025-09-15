import glob, os, pickle, cv2
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from models.depth_model import DepthModel

#datasets
from data import nyu_dataset as nyu
from data import kitti_dataset as kitti

import torch 
import torchvision.transforms as transforms

from util import util

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

class EvalArgs():
    eigen_crop = False
    garg_crop = False
    do_kb_crop = False
    min_depth_eval = 1e-3
    max_depth_eval = 80
    log_path = ""
    def __init__(self, args):
        if args.input_type == "NYUv2":
            self.dataset = "nyu"
            self.gt_path = "/data1/NYUv2/val/experiment/"
            self.eigen_crop = True
            self.max_depth_eval = 10
        if args.input_type == "KITTI":
            self.dataset = "kitti"
            self.gt_path = "/data2/kitti_dataset/data_depth_annotated/"
            self.garg_crop = True
            self.min_depth_eval = 1e-3
            self.max_depth_eval = 80
        self.pred_path = os.path.join(args.output_path, os.path.split(args.model_path)[0].split("/")[-1]+"_"+args.guide_option)

def run_midas(input_path, output_path, model_type, opt_path, input_type="NYUv2"):
    '''
    input_path: dictionary with paths
    '''
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    elif model_type == "MiDaS":
        transform = midas_transforms.default_transform
    else:
        transform = midas_transforms.small_transform

    # loaded_opt = pickle.load(open(opt_path, 'rb')) #load saved option file
    # global opt
    # opt = opt.parse() # latest option 
    # opt.__dict__.update(loaded_opt.__dict__)
    if opt_path is None:
        head, tail = os.path.split(model_path)
        opt_path = os.path.join(head, "opt.pkl")
    opt = pickle.load(open(opt_path, 'rb'))

    #guide option
    if input_type == "NYUv2":
        valid_dataset = nyu.NYUDataset(opt,input_path,is_train=False, has_label=False, transform=transform)
    elif input_type == "KITTI":
        print("KITTI")
        valid_dataset = kitti.KITTIDataset(opt,is_train=False, has_label=False, transform=transform)
    
    valid_data_loader = DataLoader(dataset=valid_dataset, num_workers= 1, shuffle=False, batch_size=1, pin_memory=True, drop_last=False)

    output_path = os.path.join(output_path, model_type)
    os.makedirs(output_path, exist_ok=True)
    
    print("start processing")

    for batch_idx, batch in tqdm(enumerate(valid_data_loader)):
        _, h,w = batch['orig_depth'].shape
        # compute
        with torch.no_grad():
            target = model(batch['rgb'].to(device)).unsqueeze(0)
            prediction = (
                torch.nn.functional.interpolate(
                    target,
                    size=(h,w),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # output
        img_name = batch['filename'][0]

        if input_type == "KITTI":
            names = img_name.split(os.sep)[4:-1]
            sub_dir = os.sep.join(names)
            output_path_ = os.path.join(output_path,sub_dir)
            os.makedirs(output_path_, exist_ok=True)
        else:
            output_path_ = output_path

        filename = os.path.join(
            output_path_, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.write_depth(filename, prediction, bits=2)

    print("finished")

def run(input_path, output_path, model_path, input_type="NYUv2", opt_path=None, guide_option="20_0", seed=2344):
    '''
    input_path: dictionary with paths
    '''
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    
    #load option
    if opt_path is None:
        head, tail = os.path.split(model_path)
        opt_path = os.path.join(head, "opt.pkl")
    
    loaded_opt = pickle.load(open(opt_path, 'rb')) #load saved option file
    global opt
    opt = opt.parse() # latest option 
    opt.__dict__.update(loaded_opt.__dict__)

    opt.ckpt = model_path
    opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0 if opt.no_instance else 1)
    opt.isTrain = False

    #load model
    model = DepthModel(opt)
    model.to(device)
    model.eval()

    if opt.experiment in ["DPT_Large"]:
        transform = transforms.Compose([
            lambda img: img / 255.0,
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    elif opt.encoder == 'MiDaS' or opt.experiment in ["MiDaS"]:
        transform = transforms.Compose([
            lambda img: img / 255.0,
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
    else:
        transform = transforms.Compose([
            lambda img: img / 255.0,
            transforms.ToTensor()
                                        ])
    #guide option
    stroke_num, use_label = guide_option.split("_")
    opt.max_stroke = int(stroke_num)
    use_label = int(use_label)
    set_seed(seed)
    
    if input_type == "NYUv2":
        valid_dataset = nyu.NYUDataset(opt,input_path,is_train=False, has_label=use_label, transform=transform)
    elif input_type == "KITTI":
        valid_dataset = kitti.KITTIDataset(opt,is_train=False, has_label=use_label, transform=transform, use_raw_depth=True)

    valid_data_loader = DataLoader(dataset=valid_dataset, num_workers= 1, shuffle=False, batch_size=1, pin_memory=True, drop_last=False)

    output_path = os.path.join(output_path, os.path.split(model_path)[0].split("/")[-1]+"_"+guide_option)
    
    os.makedirs(output_path, exist_ok=True)
    
    print("start processing")

    for batch_idx, batch in tqdm(enumerate(valid_data_loader)):
        _, h,w = batch['orig_depth'].shape
        # compute
        with torch.no_grad():
            loss, targets = model(batch)
            prediction = (
                torch.nn.functional.interpolate(
                    targets[0],
                    size=(h,w),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # output
        img_name = batch['filename'][0]
        
        if input_type == "KITTI":
            names = img_name.split(os.sep)[4:-1]
            sub_dir = os.sep.join(names)
            output_path_ = os.path.join(output_path,sub_dir)
            os.makedirs(output_path_, exist_ok=True)
        else:
            output_path_ = output_path

        filename = os.path.join(
            output_path_, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.write_depth(filename, prediction, bits=2)

    print("finished")


def prepare_run(input_path, output_path, model_path, input_type="NYUv2", opt_path=None, guide_option="20_0", seed=2344):
    '''
    input_path: dictionary with paths
    '''
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    
    #load option
    if opt_path is None:
        head, tail = os.path.split(model_path)
        opt_path = os.path.join(head, "opt.pkl")
    
    loaded_opt = pickle.load(open(opt_path, 'rb')) #load saved option file
    global opt
    # opt = opt.parse() # latest option 
    opt.__dict__.update(loaded_opt.__dict__)

    opt.ckpt = model_path
    opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0 if opt.no_instance else 1)
    opt.isTrain = False

    #load model
    model = DepthModel(opt)
    model.to(device)
    model.eval()

    if opt.encoder == 'MiDaS' or opt.experiment in ["MiDaS"]:
        transform = transforms.Compose([
            lambda img: img / 255.0,
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
    elif opt.experiment in ["DPT_Large"]:
        transform = transforms.Compose([
            lambda img: img / 255.0,
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    else:
        transform = transforms.Compose([
            lambda img: img / 255.0,
            transforms.ToTensor()
                                        ])
    #guide option
    stroke_num, use_label = guide_option.split("_")
    opt.max_stroke = int(stroke_num)
    use_label = int(use_label)
    
    #set seed for dataset
    set_seed(seed)    
    if input_type == "NYUv2":
        valid_dataset = nyu.NYUDataset(opt,input_path,is_train=False, has_label=use_label, transform=transform)
    elif input_type == "KITTI":
        valid_dataset = kitti.KITTIDataset(opt,is_train=False, has_label=use_label, transform=transform)

    valid_data_loader = DataLoader(dataset=valid_dataset, num_workers= 1, shuffle=False, batch_size=1, pin_memory=True, drop_last=False)

    return model, valid_data_loader

from eval_with_pngs import eval
def run(input_path, output_path, model_path, input_type="NYUv2", \
        opt_path=None, guide_option="20_0", seed=2344, is_live=False, eval_args=None, save_stroke=False, perturb_stroke=False):
    model, valid_data_loader = prepare_run(input_path, output_path, model_path,
                                            input_type, opt_path, 
                                            guide_option, seed)
    if is_live:
        pred_depths = []
        pred_filenames = []
        gt_depths = []

    else: # prepare to save file
        output_path = os.path.join(output_path, os.path.split(model_path)[0].split("/")[-1]+"_"+guide_option)
        if perturb_stroke:
            output_path = output_path.replace("midas","perturb_midas")
        os.makedirs(output_path, exist_ok=True)
        
    import util.util as util
    label_model = util.SenFormerInference()
    print("(run)    Start processing")
    indices=[]
    for batch_idx, batch in tqdm(enumerate(valid_data_loader)):
        _, h,w = batch['orig_depth'].shape
        # def get_luminance(img):
        #     img = img/255.0
        #     lum = (0.2126 * img[:,:,0]).mean() + (0.7152 * img[:,:,1]).mean() + (0.0722 * img[:,:,2]).mean()
        #     return lum
        # orig_rgb = batch['orig_rgb'].squeeze().cpu().numpy()
        # import PIL.Image as Image
        # name_ = os.path.split(batch['filename'][0])[-1].split(".")[0]
        # if get_luminance(orig_rgb) > 0.50: continue
        # else: Image.fromarray(orig_rgb.astype(np.uint8)).save("/personal/JungEunYoo/low_resolution/{}.png".format(name_))

        # compute
        with torch.no_grad():
            if perturb_stroke and int(guide_option.split("_")[0]) > 0:
                guide_layer = util.perturb_stroke(batch['guide'].squeeze().numpy())
                batch['guide'] = torch.tensor(guide_layer).unsqueeze(0).unsqueeze(0)
    
            loss, targets = model(batch)
            prediction = (
                torch.nn.functional.interpolate(
                    targets[0],
                    size=(h,w),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
                # output
            img_name = batch['filename'][0]
            
            if input_type == "KITTI":
                names = img_name.split(os.sep)[4:-1]
                sub_dir = os.sep.join(names)
                output_path_ = os.path.join(output_path,sub_dir)
                os.makedirs(output_path_, exist_ok=True)
            else:
                output_path_ = output_path

            filename = os.path.join(
                output_path_, os.path.splitext(os.path.basename(img_name))[0]
            )
            # # numerical name
            # filename = os.path.join(
            #     output_path_, str(batch_idx))

            if is_live:
                max_val = (2**(8*2))-1
                prediction = max_val * (prediction-prediction.min())/(prediction.max()-prediction.min())
                pred_depths.append(prediction)
                pred_filenames.append(filename)
            else:
                # import PIL.Image as Image
                # Image.fromarray(batch['orig_rgb'].detach().cpu().squeeze().numpy().astype(np.uint8)).save(filename+".jpg")
                util.write_depth(filename, prediction, bits=2)
                if save_stroke:
                    util.write_depth(filename+"_stroke",batch['guide'].cpu().squeeze().numpy(),bits=1)
    # get gt depths
    print(indices)
    if is_live:
        missing_ids = set()
        if input_type == 'KITTI':
            for t_id in range(len(pred_depths)):
                file_dir = '/'.join(pred_filenames[t_id].split('/')[-5:]) #filename without extention
                gt_depth_path = os.path.join(eval_args.gt_path,file_dir) + ".png"
                depth = cv2.imread(gt_depth_path, -1)
                if depth is None:
                    print('Missing: %s ' % gt_depth_path)
                    missing_ids.add(t_id)
                    continue

                depth = depth.astype(np.float32) / 256.0
                gt_depths.append(depth)
                
                '''for rtdd'''
                '''save gt file for eval'''
                # target_path = "/personal/JungEunYoo/rtdd_kitti/gt/"
                # import shutil
                # shutil.copyfile(gt_depth_path, target_path+"{}.png".format(t_id))
       
        elif input_type == 'NYUv2':
            for t_id in range(len(pred_depths)):
                file_dir = pred_filenames[t_id].split('/')[-1]
                filename = file_dir.split('_')[-1]
                gt_depth_path = os.path.join(eval_args.gt_path, 'sync_depth_' + filename + '.png')
                depth = cv2.imread(gt_depth_path, -1)
                if depth is None:
                    print('Missing: %s ' % gt_depth_path)
                    missing_ids.add(t_id)
                    continue

                depth = depth.astype(np.float32) / 1000.0
                gt_depths.append(depth)

        print('GT files reading done')
        print('{} GT files missing'.format(len(missing_ids)))

        print('Computing errors')
        eval(eval_args,pred_depths,gt_depths, missing_ids, pred_filenames)

        print('Done')

    else:
        print("Finished saving inferred depth")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', 
        default='input',
        help='folder with input images'
    )

    parser.add_argument('-o', '--output_path', 
        default='output',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model_path', 
        default=None,
        help='path to the trained weights of model'
    )

    parser.add_argument('--opt_path',
        default = None,
        help = "path to option file")
    
    parser.add_argument('--guide_option',
        default = "20_0",
        help = "number of strokes and using label (0,-1)")

    parser.add_argument('--input_type',
        default = "NYUv2",
        help = "NYUv2 or KITTI")
    parser.add_argument('--set_seed', type=int, default = 2344)
    parser.add_argument('--is_live', action='store_true')
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--save_stroke', action='store_true')
    parser.add_argument('--perturb_stroke', action='store_true')

    from options.test_options import TestOptions
    opt = TestOptions()
    parser = opt.initialize(parser)
    opt.parser = parser
    opt = opt.parse() # latest option 

    args = parser.parse_args()

    if args.model_path in ['MiDaS', "DPT_Large", "DPT_Hybrid"]:
        run_midas(args.input_path,
                args.output_path, 
                args.model_path, 
                args.opt_path, 
                input_type=args.input_type)
    else:
        if args.is_live:
            eval_args = EvalArgs(args)
            eval_args.log_path = eval_args.pred_path + "_log.csv"
            for i in range(args.iter):
                eval_args.seed = args.set_seed + i #add seed information
                run(args.input_path, 
                    args.output_path, 
                    args.model_path, 
                    guide_option=args.guide_option, 
                    seed=args.set_seed + i,
                    input_type=args.input_type,
                    is_live = True,
                    eval_args = eval_args,
                    perturb_stroke=args.perturb_stroke)
        else:
            run(args.input_path, 
                args.output_path, 
                args.model_path, 
                guide_option=args.guide_option, 
                seed=args.set_seed,
                input_type=args.input_type,
                save_stroke=args.save_stroke,
                perturb_stroke=args.perturb_stroke)
        # eval_args = EvalArgs(args)
        # import eval_with_pngs
        # eval_with_pngs.main(eval_args)



'''
run script

#ours
###NYU
python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/s3r_midas/latest_net_G.pth" \
--guide_option "0_0" --guide_empty 0.0 --is_live --perturb_stroke

python run.py --input_path "/data1/NYUv2/" \
--output_path "/data1/jey/output/" --model_path "checkpoints/midasE_R_LL_testD_guide0-1/latest_net_G.pth" \
--guide_option "0_0" --guide_empty 0.0 --is_live

python run.py --input_path "/data1/NYUv2/" \
--output_path "/data1/jey/output/" --model_path "checkpoints/dpt/latest_net_G.pth" \
--guide_option "20_0"

#for multiple iteration
python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" --model_path "checkpoints/midasE_R_LL_testD_guide0-1/latest_net_G.pth" --guide_option "0_0" --guide_empty 0.0 --is_live --iter 5
python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/midasE_LL_testD_guide0-1/latest_net_G.pth" --guide_option "0_0" \
--guide_empty 0.0 --is_live --iter 1 --set_seed 2345

# for saving stroke for comparison
python run.py --input_type KITTI \
--output_path "/personal/JungEunYoo/rtdd_kitti/" --model_path "checkpoints/midasE_R_LL_testD_guide0-1_384/latest_net_G.pth" \
--guide_option "0_0" --load_size 512 --save_stroke

###Kitti
python run.py --input_type KITTI --output_path "/data1/jey/output_kitti/" \
--model_path "checkpoints/midasE_R_LL_testD_guide0-1_384/latest_net_G.pth" \
--guide_option "20_0" --guide_empty 0.0 --load_size 512 --is_live --perturb_stroke

python run.py --input_type KITTI --output_path "/data1/jey/output_kitti/" \
--model_path "checkpoints/midasE_R_LL_testD_guide0-1/latest_net_G.pth" --guide_option "5_0" \
--guide_empty 0.0 --is_live --iter 5

#midas
python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" --model_path MiDaS --opt_path "checkpoints/midasE_R_LL_testD331_guide0-1/latest_net_G.pth"
#dpt
python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" --model_path DPT_Large --opt_path "checkpoints/midasE_R_LL_testD331_guide0-1/opt.pkl"

###KITTI
python run.py --output_path "/data1/jey/output_kitti/" --model_path DPT_Large --opt_path "checkpoints/midasE_R_LL_testD331_guide0-1/opt.pkl" --input_type KITTI

'''
