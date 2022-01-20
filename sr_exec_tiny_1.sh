n_gpu=0
config_file=configs/sr/tiny_sr.yml

python main.py \
    --config $config_file \
    --smart_ratio 0.995 \
    --subfolder sc_tiny_995 \
    --gpu $n_gpu

