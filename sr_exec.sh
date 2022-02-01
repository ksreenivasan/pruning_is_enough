config_file="configs/sr/resnet20/resnet20_sr.yml"
n_gpu=1
subfolder=check_sr_sparsity

# 0.59
python main.py \
    --config $config_file \
    --smart_ratio 0.9941 \
    --subfolder $subfolder \
    --gpu $n_gpu

# 1.44
python main.py \
    --config $config_file \
    --smart_ratio 0.9856 \
    --subfolder $subfolder \
    --gpu $n_gpu

# 3.72
python main.py \
    --config $config_file \
    --smart_ratio 0.9628 \
    --subfolder $subfolder \
    --gpu $n_gpu

# 5
python main.py \
    --config $config_file \
    --smart_ratio 0.95 \
    --subfolder $subfolder \
    --gpu $n_gpu

# 10
python main.py \
    --config $config_file \
    --smart_ratio 0.9 \
    --subfolder $subfolder \
    --gpu $n_gpu

# 20
python main.py \
    --config $config_file \
    --smart_ratio 0.8 \
    --subfolder $subfolder \
    --gpu $n_gpu

# 50
python main.py \
    --config $config_file \
    --smart_ratio 0.5 \
    --subfolder $subfolder \
    --gpu $n_gpu
