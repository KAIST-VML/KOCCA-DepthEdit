'''depth model - input: RGB+stroke output: depth'''
CUDA_VISIBLE_DEVICES=1 python train.py --name attn_5e-5_B16 \
--train_mode depth --netG depth  \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 16 --nThreads 8 \
--niter 30 --valid_size 16 --use_rgb --no_label_encoder --mobilenet --max_stroke 5
--input_ch 4


'''depth model segguide'''
CUDA_VISIBLE_DEVICES=3 python train.py --name segguideFreezeOnlyEncoder_normalize_stroke_mobilenetEncoder_megadepth_10+4_30epoch \
--train_mode depth --netG segDepth  \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet

# sum feature
CUDA_VISIBLE_DEVICES=2,3 python train.py --name segguide_sum_feature_mobilenetEncoder_megadepth_10+4_30epoch \
--train_mode depth --netG segDepth  \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --no_guide

'''depth model - continuity loss'''
CUDA_VISIBLE_DEVICES=1 python train.py --name segguide_stroke_edge008_mobilenetEncoder_megadepth_10+4_30epoch \
--train_mode depth --netG segDepth  \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --force_edge

'''depth with spade decoder'''
CUDA_VISIBLE_DEVICES=0 python train.py --name ADE_SPADE \
--train_mode depth --netG segDepth  \
--dataset_mode ade20k --dataroot /data1/ADEChallengeData2016/training/ --no_instance --segmap_type ade20k \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --decoder spade

'''segguide: use coco-stuff'''
CUDA_VISIBLE_DEVICES=1 python train.py --name segguideCoco_spade135BeforeCat-1.0_stroke_mobilenetEncoder_10+5_30epoch_data1grad05_B16 \
--train_mode depth --netG segDepth  \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 16 --nThreads 8 \
--niter 30 --valid_size 16 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff

'''segguide: use coco-stuff / no skip connection at up 4 and 5'''
CUDA_VISIBLE_DEVICES=0 python train.py --name noSkipAtUp4n5_max10 \
--train_mode depth --netG segDepth \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 16 --nThreads 8 \
--niter 30 --valid_size 16 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 10

'''segguide: use coco-stuff / from 0 to 5 strokes / input channel 5'''
CUDA_VISIBLE_DEVICES=3 python train.py --name stroke0to5_modifyInputCH_ADE \
--train_mode depth --netG segDepth  \
--dataset_mode ade20k --dataroot /data1/ADEChallengeData2016/training/ --no_instance \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 16 --nThreads 8 \
--niter 30 --valid_size 16 --use_rgb --no_label_encoder --mobilenet \
--segmap_type coco-stuff --input_ch 4

'''segguide: use coco-stuff / from 0 to 5 strokes / input channel 4 / single'''
CUDA_VISIBLE_DEVICES=0 python train.py --name stroke0to5_modifyInputCH_ADE_single \
--train_mode depth --netG segDepth  \
--dataset_mode ade20k --dataroot ./datasets/single/ADE/ --no_instance --single_data \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 1 \
--niter 3000000 --use_rgb --no_label_encoder --mobilenet \
--segmap_type coco-stuff --input_ch 4

'''--------------depth: refine-------------------'''
CUDA_VISIBLE_DEVICES=2 python train.py --name refineDepth \
--train_mode depth --netG depth  \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 16 --nThreads 8 \
--niter 30 --valid_size 16 --use_rgb --no_label_encoder --mobilenet --refine_depth

'''depth: refine with refined input channel'''
CUDA_VISIBLE_DEVICES=3 python train.py --name refineDepth_modifyInputCH \
--train_mode depth --netG depth  \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 16 --nThreads 16 \
--niter 30 --valid_size 16 --use_rgb --no_label_encoder --mobilenet --refine_depth

python train.py --name refine_deblur \
--train_mode depth --netG depth  \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 16 --use_rgb --no_label_encoder --mobilenet --refine_depth --debug


CUDA_VISIBLE_DEVICES=0 python train.py --name sharpen_testD330 \
--train_mode depth --netG segDepth --encoder MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke -1 \
--input_ch 4 --random_label --test_decoder --guide_empty 0.0 --debug
'''-----------------------------------------------------'''
''' SPADE Decoder '''
CUDA_VISIBLE_DEVICES=2 python train.py --name spade_B4_2 \
--train_mode depth --netG segDepth --encoder MobileNetV2 --decoder spade \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --seed 222

'''segguide: use coco-stuff / no skip connection at up 4 and 5 '''
CUDA_VISIBLE_DEVICES=1 python train.py --name strokeEnc_up5 \
--train_mode depth --netG segDepth --encoder MobileNetV2 \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 


"------------------------------"

'''baseline'''
CUDA_VISIBLE_DEVICES=1 python train.py --name base_B8 \
--train_mode depth --netG segDepth --encoder MobileNetV2 \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 8 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --debug

'''midas baseline'''
CUDA_VISIBLE_DEVICES=2 python train.py --name noSPADE_att \
--train_mode depth --netG segDepth --encoder MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --debug

'''midas baseline w/ ADE20k'''
CUDA_VISIBLE_DEVICES=2 python train.py --name midasE_ade \
--train_mode depth --netG segDepth --encoder MiDaS \
--dataset_mode ade20k --dataroot /data1/ADEChallengeData2016/training/ \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 150 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --debug --continue_train --ckpt "checkpoints/midasE_ade/latest_net_G.pth"

'''midas - no stroke'''
CUDA_VISIBLE_DEVICES=2 python train.py --name midasE_noStroke \
--train_mode depth --netG segDepth --encoder MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 3 \
--input_ch 3 --debug --no_guide

'''Midas No Spade'''
CUDA_VISIBLE_DEVICES=2 python train.py --name noSPADE_att \
--train_mode depth --netG depth --encoder MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --max_stroke 20 \
--input_ch 4 --debug --123


'''midas encoder'''
CUDA_VISIBLE_DEVICES=2 python train.py --name testttttt \
--train_mode depth --netG segDepth --encoder MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss --local_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --random_label

#with spade
CUDA_VISIBLE_DEVICES=0 python train.py --name midasE_spadeD \
--train_mode depth --netG segDepth --encoder MiDaS --decoder spade \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss --local_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --guide_empty 0.0 --debug

#with test decoder
CUDA_VISIBLE_DEVICES=0 python train.py --name midasE_R_LL_testD_guide0-1_384 \
--train_mode depth --netG segDepth --encoder MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss --local_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --random_label --test_decoder --guide_empty 0.0 --debug

CUDA_VISIBLE_DEVICES=3 python train.py --name midasEnc_RL_LL_384_2 \
--train_mode depth --netG segDepth --encoder MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss --local_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --random_label --load_size 400 --crop_size 384 \
--ckpt "checkpoints/midasEnc_RL_LL_384/6_net_G.pth"

CUDA_VISIBLE_DEVICES=2 python train.py --name midasEnc_att2 \
--train_mode depth --netG segDepth --encoder MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --debug

#refine with Midas
CUDA_VISIBLE_DEVICES=2 python train.py --name RefineDepth_test22 \
--train_mode depth --netG segDepth --encoder MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --debug --ckpt  "/data1/jey/DepthEdit_logs/checkpoints/midasEnc_randomLabel_LocalLoss/latest_net_G.pth"

'''----------------------------------Experiment------------------------------------'''
CUDA_VISIBLE_DEVICES=3 python train.py --name midas \
--train_mode depth --experiment MiDaS \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 

#dpt / 256
python train.py --name dpt_norm05 \
--train_mode depth --experiment DPT-Large \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --debug --load_size 400 --crop_size 384 --guide_empty 0.0