
# SRv1: original SR (in Jason Lee's paper)
#config_file="configs/sr/resnet20/resnet20_sr.yml"
#subfolder=tmp

# SRv2: original SR + change 1st/last layer sparsity (follow GM's one)
:<<BLOCK
config_file="configs/sr/resnet20/resnet20_srV2.yml"
n_gpu=2
subfolder=SRv2_sp_1_44_debug
# 1.44% sparsity
python main.py \
    --config $config_file \
    --smart_ratio 0.9856 \
    --subfolder $subfolder \
    --gpu $n_gpu
# 3.72% sparsity
#python main.py \
#    --config $config_file \
#    --smart_ratio 0.9628 \
#    --subfolder $subfolder \
#    --gpu $n_gpu
BLOCK


# SRv3: Start from SRv2, finetune p_i for each layer
## Step 1: fine-tune ratio
:<<BLOCK
config_file="configs/sr/resnet20/resnet20_find_srV3.yml"
n_gpu=1
subfolder=find_SRv3_sp_1_44_debug_lr_1e-6
python main.py \
    --config $config_file \
    --target-sparsity 1.44 \
    --subfolder $subfolder \
    --gpu $n_gpu
BLOCK

## Step 2: train the model from the obtained smart ratio
:<<BLOCK
conf_file="configs/sr/resnet20/resnet20_srV3.yml"
log_root="srV3_1e-6_real_"
log_end="_log"
subfolder_root="srV3_1e-6_real_"

for epoch in 10 30 70 160
do
    python main.py \
    --config "$conf_file" \
    --smart_ratio 0.9856 \
    --srV3-epoch $epoch \
    --subfolder "$subfolder_root$epoch" > "$log_root$epoch$log_end" 2>&1 &
done
BLOCK


# SRv4: original SR + change 1st/last layer sparsity to 100%
#:<<BLOCK
config_file="configs/sr/resnet20/resnet20_srV4.yml"
n_gpu=2
subfolder=SRv4_sp_1_44
# 1.44% sparsity
python main.py \
    --config $config_file \
    --smart_ratio 0.9856 \
    --subfolder $subfolder \
    --gpu $n_gpu
#BLOCK
