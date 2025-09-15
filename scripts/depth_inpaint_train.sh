# CUDA_VISIBLE_DEVICES=0 python train.py --name seg2depth_gpu0 --dataset_mode coco --dataroot /mnt/data2/coco-stuff --no_vgg_loss --megadepth_loss --no_ganFeat_loss --niter 10

#spade model
CUDA_VISIBLE_DEVICES=0 python train.py --name seg2deph_test333 --dataset_mode coco \
--dataroot /mnt/data2/coco-stuff --no_vgg_loss --megadepth_loss \
--no_ganFeat_loss --niter 1

CUDA_VISIBLE_DEVICES=0 python train.py --name seg2depth_gpu0 \
--dataset_mode coco --dataroot /mnt/data2/coco-stuff --no_vgg_loss \
--l1_loss --no_ganFeat_loss --niter 10

#seg2depth model
CUDA_VISIBLE_DEVICES=0 python train.py --name mask_test_reconloss --netG seg2depth \
--dataset_mode coco --dataroot /mnt/data2/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--niter 1 --valid_size 10 --mask ./datasets/mask/mask.jpg


#rgb + segmap
CUDA_VISIBLE_DEVICES=0 python train.py --name rgb_test --netG seg2depth \
--dataset_mode coco --dataroot /mnt/data2/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--niter 1 --valid_size 10 --use_rgb

#rgb + segmap w/ separate encoder
CUDA_VISIBLE_DEVICES=6 python train.py --name independent_rgb_local_epoch10 --netG seg2depth \
--dataset_mode coco --dataroot ../data/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--niter 10 --valid_size 10 --use_rgb --no_label_encoder

#rgb + segmap w/ separate encoder (single)
CUDA_VISIBLE_DEVICES=0 python train.py --name independent_rgb_local_epoch10 --netG seg2depth \
--dataset_mode coco --dataroot ./datasets/single \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--niter 10 --use_rgb --no_label_encoder

#masked depth + segmap w/ separate encoder
CUDA_VISIBLE_DEVICES=0 python train.py --name mask_megadepth_10+5_seg --netG seg2depth \
--dataset_mode coco --dataroot /mnt/data2/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.00001 \
--niter 10 --valid_size 10 --mask ./datasets/mask/mask.jpg --no_label_encoder

#debug
CUDA_VISIBLE_DEVICES=1 python train.py --name only_megadepth_10+4_seg --netG seg2depth \
--dataset_mode coco --dataroot /mnt/data2/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 \
--niter 10 --valid_size 10 --mask ./datasets/mask/mask.jpg --no_label_encoder

#diceloss
CUDA_VISIBLE_DEVICES=1 python train.py --name dice_megadepth_10+4_seg --netG seg2depth \
--dataset_mode coco --dataroot /mnt/data2/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss --dice_loss \
--lr 0.0001 \
--niter 10 --valid_size 10 --mask ./datasets/mask/mask.jpg --no_label_encoder

#rgb+midas
CUDA_VISIBLE_DEVICES=2 python train.py --name rgb_midas_10+4_seg --netG seg2depth \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --midas_loss \
--lr 0.0001 \
--niter 10 --valid_size 10 --use_rgb --no_label_encoder --no_guide

#mask+midas
CUDA_VISIBLE_DEVICES=1 python train.py --name mask_midas_l1_10+4_seg --netG seg2depth \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --midas_loss \
--lr 0.0001 \
--niter 10 --valid_size 10 --mask ./datasets/mask/mask.jpg --no_label_encoder
