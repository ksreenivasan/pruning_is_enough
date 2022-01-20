n_gpu=1
config_file=configs/sr/tiny_sr.yml

python main.py \
    --config $config_file \
    --smart_ratio 0.5 \
    --subfolder sc_tiny_50 \
    --gpu $n_gpu


python main.py \
    --config $config_file \
    --smart_ratio 0.8 \
    --subfolder sc_tiny_80 \
    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.95 \
    --subfolder sc_tiny_95 \
    --gpu $n_gpu


python main.py \
    --config $config_file \
    --smart_ratio 0.986 \
    --subfolder sc_tiny_986 \
    --gpu $n_gpu

