python train.py --name dpt \
--train_mode depth --experiment DPT-Large \
--dataset_mode coco --dataroot /data1/coco-stuff \
--no_vgg_loss --no_ganFeat_loss --megadepth_loss \
--lr 0.0001 --batchSize 4 --nThreads 8 \
--niter 30 --valid_size 10 --use_rgb --no_label_encoder --mobilenet --segmap_type coco-stuff --max_stroke 20 \
--input_ch 4 --debug