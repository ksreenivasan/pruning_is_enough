n_gpu=0
config_file=configs/sr/wideresnet_sr.yml

python main.py \
    --config $config_file \
    --smart_ratio 0.98 \
    --subfolder sc_wide_98 \
    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.8 \
    --subfolder sc_wide_80 \
    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.5 \
    --subfolder sc_wide_50 \
    --gpu $n_gpu


python main.py \
    --config $config_file \
    --smart_ratio 0.9 \
    --subfolder sc_wide_90 \
    --gpu $n_gpu
