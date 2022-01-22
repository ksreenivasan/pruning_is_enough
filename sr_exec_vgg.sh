n_gpu=1
config_file=configs/sr/vgg_sr.yml

python main.py \
    --config $config_file \
    --smart_ratio 0.95 \
    --subfolder sc_vgg_95_correct \
    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.983 \
    --subfolder sc_vgg_983_correct \
    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.5 \
    --subfolder sc_vgg_50_correct \
    --gpu $n_gpu


python main.py \
    --config $config_file \
    --smart_ratio 0.9 \
    --subfolder sc_vgg_90_correct \
    --gpu $n_gpu

