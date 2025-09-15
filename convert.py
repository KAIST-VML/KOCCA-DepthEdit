# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# This code was checked to work with the following library versions:
#
# ONNX-TensorFlow:  1.7.0   [pip install onnx-tf==1.7.0]
# ONNX:             1.8.0   [pip install onnx==1.8.0]
# TensorFlow:       2.4.0   [pip install tensorflow==2.4.0]
# PyTorch:          1.7.1   [pip install ]
#
# More information about ONNX-TensorFlow: https://github.com/onnx/onnx-tensorflow

import torch.nn as nn
import torch
import os
from models.depth_model import DepthModel
from options.test_options import TestOptions

# DO NOT COMMENT THIS LINE (IT IS DISABLING GPU)!
# WHEN COMMENTED, THE RESULTING TF MODEL WILL HAVE INCORRECT LAYER FORMAT
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import onnx
from onnx_tf.backend import prepare
from collections import OrderedDict
import tensorflow as tf

# python convert.py --ckpt "/data1/jey/DepthEdit_logs/checkpoints/midasEnc_randomLabel_LocalLoss/latest_net_G.pth" --gpu_ids -1 --netG segDepth --segmap_type coco-stuff --input_ch 4 --encoder MiDaS
# python convert.py --ckpt "D:\Dropbox\01_Work\VML\03_Research\03_Seg2Depth\source\DepthEdit\ckpt\segDepth\midas.pth" --gpu_ids -1 --netG segDepth --experiment MiDaS --segmap_type coco-stuff --input_ch 4 --encoder MiDaS
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert option")
    # parser.add_argument("--model",
    #                     type=str,
    #                     help="Model type",
    #                     default='mobilenet')
    parser.add_argument("--save_dir",
                        type=str,
                        help="save directory path",
                        default='./tf_model/')
    parser.add_argument("--save_name",
                        type=str,
                        help="converted model name",
                        default='model')
    parser.add_argument("--pretrained",
                        help="if set disables augment",
                        action="store_true")
    parser.add_argument("--load_dir",
                        type=str,
                        help="load directory path",
                        default='tf_model/')
    parser.add_argument("--img_size",
                        type=int,
                        help="image size",
                        default=256)
   
    # configs = parser.parse_args()
    opt = TestOptions()
    opt.parser = opt.initialize(parser)
    opt = opt.parse() 
    
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    # parse name
    pb_model_name = os.path.join(opt.save_dir, opt.save_name) +".pb"
    tflite_model_name = os.path.join(opt.save_dir, opt.save_name) +".tflite"
    onnx_model_name = os.path.join(opt.save_dir,  opt.save_name) + ".onnx"
    
    # if opt.pretrained: #if loading pretrained model
    #     state_dict = torch.load(opt.load_dir)['model']
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:] # remove `module.`
    #         new_state_dict[name] = v
    #     # load params
    #     print(new_state_dict)
    #     model.load_state_dict(new_state_dict)
    
        
    # Creating / loading model   
    
    
    depth_model = DepthModel(opt)
    depth_model.eval()

    model = depth_model.model

    # Converting model to ONNX

    for _ in model.modules():
        _.training = False
    
    sample_input = torch.randn(1, 4, opt.img_size, opt.img_size)
    input_nodes = ['input']
    output_nodes = ['output']
    
    import pdb; pdb.set_trace();
    torch.onnx.export(model, sample_input, onnx_model_name, export_params=True, input_names=input_nodes, output_names=output_nodes)
    print(">>>>> Sucessfully export to onnx model <<<<<")
    # Converting model to Tensorflow

    onnx_model = onnx.load(onnx_model_name)
    output = prepare(onnx_model)
    pdb.set_trace()
    output.export_graph(opt.save_dir) #saved in the save_dir as "saved_model.pb"
    print(">>>>> Sucessfully export to tf model <<<<<")
    # Exporting the resulting model to TFLite

    converter = tf.lite.TFLiteConverter.from_saved_model(opt.save_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    
    tflite_model = converter.convert()
    open(tflite_model_name, "wb").write(tflite_model)