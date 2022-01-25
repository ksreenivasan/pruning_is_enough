config_file="configs/sr/resnet20/resnet20_sr.yml"
n_gpu=1

#python main.py \
#    --config $config_file \
#    --smart_ratio 0.9 \
#    --subfolder sc_resnet20_90 \
#    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.8 \
    --subfolder sc_resnet20_80 \
    --gpu $n_gpu

python main.py \
    --config $config_file \
    --smart_ratio 0.5 \
    --subfolder sc_resnet20_50 \
    --gpu $n_gpu
