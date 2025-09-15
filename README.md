# KOCCA-DepthEdit


This repository contains code for Interactive Depth Estimation, allowing you to generate depth maps interactively.

## Requirements

- Python 3.8
- PyTorch 1.12.0
- CUDA 11.2 (recommended)

## Run Application

```bash
python interface.py --gpu_ids 0 --netG segDepth --segmap_type coco-stuff --input_ch 4 --test_decoder --encoder MiDaS --guide_empty 0.0
