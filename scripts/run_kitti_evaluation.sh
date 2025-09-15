#this is for evaulatoin

# $1: model name
# $2: guide_option
eval=${3:-2344}

if [ "$eval" != 'eval' ]
then
python run.py --input_type KITTI --output_path "/data1/jey/output_kitti/" --model_path "checkpoints/$1/latest_net_G.pth" --guide_option $2 --guide_empty 0.0 --set_seed $eval
fi
python eval_with_pngs.py --pred_path "/data1/jey/output_kitti/$1_$2/" --gt_path "/data2/kitti_dataset/data_depth_annotated/" --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --garg_crop
