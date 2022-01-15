n_gpu=0
config_file=configs/sr/vgg_sr.yml

python main.py \
    --config $config_file \
    --smart_ratio 0.98 \
    --subfolder sc_vgg_98 \
    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.8 \
    --subfolder sc_vgg_80 \
    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.5 \
    --subfolder sc_vgg_50 \
    --gpu $n_gpu


python main.py \
    --config $config_file \
    --smart_ratio 0.9 \
    --subfolder sc_vgg_90 \
    --gpu $n_gpu
