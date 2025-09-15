python interface.py --gpu_ids 0 --netG segDepth --segmap_type coco-stuff --input_ch 4 --test_decoder --encoder MiDaS --guide_empty 0.0

# for mac (mps)
python interface.py --gpu_ids -1 --netG segDepth --segmap_type coco-stuff --input_ch 4 --test_decoder --encoder MiDaS --guide_empty 0.0
