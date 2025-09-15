CUDA_VISIBLE_DEVICES=2 python train.py --name multi_stroke0to5_ADE_005_B4 \
--train_mode depth --netG segDepth  --multi_head \
--dataset_mode ade20k --dataroot /data1/ADEChallengeData2016/training/ --no_instance \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 4 --nThreads 8 \
--niter 150 --valid_size 16 --use_rgb --no_label_encoder --mobilenet \
--input_ch 3

CUDA_VISIBLE_DEVICES=1 python train.py --name multi_attn-depth_seg-_coco \
--train_mode depth --netG segDepth  --multi_head \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 16 --use_rgb --no_label_encoder --mobilenet \
--input_ch 3

'''single'''
CUDA_VISIBLE_DEVICES=1 python train.py --name multi_stroke0to5_modifyInputCH_ADE_single \
--train_mode depth --netG segDepth  --multi_head \
--dataset_mode ade20k --dataroot ./datasets/single/ADE/ --no_instance --single_data \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00005 --batchSize 1 \
--niter 3000000 --use_rgb --no_label_encoder --mobilenet \
--input_ch 3