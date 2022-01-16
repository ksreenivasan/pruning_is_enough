n_gpu=0
config_file=configs/sr/vgg_sr.yml

python main.py \
    --config $config_file \
    --smart_ratio 0.95 \
    --subfolder sc_vgg_95 \
    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.995 \
    --subfolder sc_vgg_995 \
    --gpu $n_gpu

<<:BLOCK
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

:BLOCK
