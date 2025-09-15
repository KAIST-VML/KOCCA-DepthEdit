#this is for evaulatoin

# $1: model name
# $2: guide_option
eval=${3:-2344}

if [ "$eval" != 'eval' ]
then
python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" --model_path "checkpoints/$1/latest_net_G.pth" --guide_option $2 --guide_empty 0.0 --set_seed $eval
fi
python eval_with_pngs.py --pred_path "/data1/jey/output/$1_$2/" --gt_path "/data1/NYUv2/val/experiment" --dataset nyu --max_depth_eval 10 --eigen_crop