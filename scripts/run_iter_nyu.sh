python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "0_$2" \
--guide_empty 0.0 --is_live --iter 1

python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "1_$2" \
--guide_empty 0.0 --is_live --iter $3

python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "2_$2" \
--guide_empty 0.0 --is_live --iter $3

python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "3_$2" \
--guide_empty 0.0 --is_live --iter $3

python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "4_$2" \
--guide_empty 0.0 --is_live --iter $3

python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "5_$2" \
--guide_empty 0.0 --is_live --iter $3

python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "10_$2" \
--guide_empty 0.0 --is_live --iter $3

python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "15_$2" \
--guide_empty 0.0 --is_live --iter $3

python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "20_$2" \
--guide_empty 0.0 --is_live --iter $3

python run.py --input_path "/data1/NYUv2/" --output_path "/data1/jey/output/" \
--model_path "checkpoints/$1/latest_net_G.pth" --guide_option "25_$2" \
--guide_empty 0.0 --is_live --iter $3