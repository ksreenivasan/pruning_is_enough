






######################################################
##########   ResNet-18   #############################
######################################################

# weight training
:<<BLOCK
conf_file="configs/training/resnet18/cifar10_resnet18_training.yml"
subfolder_root="resnet18_cifar10_wt_"
log_end="_log"

python main.py \
    --config "$conf_file" \
    --subfolder "$subfolder_root" > "$subfolder_root$log_end" 2>&1 &
BLOCK

# smart ratio
:<<BLOCK
conf_file="configs/sr/resnet18/cifar10_resnet18_sr.yml"
subfolder_root="resnet18_cifar10_sr_"
log_end="_log"

for sr in 0.95 0.98 0.995
do
    python main.py \
    --config "$conf_file" \
    --smart_ratio "$sr" \
    --subfolder "$subfolder_root$sr" > "$subfolder_root$sr$log_end" 2>&1 #&
done
BLOCK

# Gem-Miner (hypercube)
#:<<BLOCK

# 5% sparsity
#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_real_"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6_iter_20.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6_iter_20_multistep_0_1.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_multistep_0_1"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6_iter_20_cosine_0_1.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_cosine_0_1"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6_iter_20_adam.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_adam"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_1lam6_iter_20_adam.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_adam_1lam6"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_3lam6_iter_20_adam.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_adam_3lam6"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_1lam6_iter_20_adam_1e4.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_adam_1lam6_1e4"

conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_1lam6_iter_20_adam_1e3.yml"
subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_adam_1lam6_1e3"


# 2% sparsity
#conf_file="configs/hypercube/resnet18/cifar10/sparsity_2_9lam6_iter_20.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_2_real_"

# 0.5% sparsity
#conf_file="configs/hypercube/resnet18/cifar10/sparsity_0_5_1lam5.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_0_5_real_"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_0_5_85lam6_iter_20_adam.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_0_5_iter_20_adam_85lam6"

log_end="_log"

python main.py \
--config "$conf_file" \
--subfolder "$subfolder_root" > "$subfolder_root$log_end" 2>&1 #&
BLOCK



######################################################
##########   ResNet-20   #############################
######################################################

# turn on bias, affine batch-norm
:<<BLOCK
config_file="configs/hypercube/resnet20/bias_AffineBN_True/sparsity_1_44.yml"
subfolder_root="resnet20_bias_affine_True_hc_sparsity_1_44"
log_end="_log"

python main.py \
--config "$config_file" \
--subfolder "$subfolder_root" > "$subfolder_root$log_end" 2>&1 #&
BLOCK





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


for epoch in 10 30 70 149 #160
do
    python main.py \
    --config "$conf_file" \
    --smart_ratio 0.9856 \
    --srV3-epoch $epoch \
    --subfolder "$subfolder_root$epoch" > "$log_root$epoch$log_end" 2>&1 &
done
#BLOCK


# SRv4: original SR + change 1st/last layer sparsity to 100%
:<<BLOCK
config_file="configs/sr/resnet20/resnet20_srV4.yml"
n_gpu=2
subfolder=SRv4_sp_1_44
# 1.44% sparsity
python main.py \
    --config $config_file \
    --smart_ratio 0.9856 \
    --subfolder $subfolder \
    --gpu $n_gpu
BLOCK


# SRv5: grid search
:<<BLOCK
group=4
n_gpu=1

# step 1. do the grid search
config_file="configs/sr/resnet20/resnet20_sr_grid.yml"
subfolder=SR_grid_sp_1_44_
COUNTER=24*$group+0
input="per_layer_sparsity_resnet20/grid_search_saved_$group.csv"
while IFS= read -r line
do
	COUNTER=$((COUNTER + 1))
	echo "$COUNTER"
	echo "$line"
	python main.py \
    	--config $config_file \
    	--smart_ratio 0.9856 \
    	--subfolder $subfolder$COUNTER \
    	--gpu $n_gpu \
		--sr_seq $line
done < "$input"
BLOCK

# step 2. check the best accuracy among different runs
#python grid_search_SR.py 



# SRv6: Start from SRv5, finetune p_i for each layer

#echo "first, get SRv5 info"

## Step 1: fine-tune ratio
:<<BLOCK
config_file="configs/sr/resnet20/resnet20_find_srV6.yml"
n_gpu=1
subfolder=find_SRv6_sp_1_44_debug_lr_1e-6
python main.py \
    --config $config_file \
    --target-sparsity 1.44 \
    --subfolder $subfolder \
    --gpu $n_gpu
BLOCK




## Step 2: train the model from the obtained smart ratio
:<<BLOCK
conf_file="configs/sr/resnet20/resnet20_srV6.yml"
log_root="srV6_1e-6_real_"
log_end="_log"
subfolder_root="srV6_1e-6_real_"


for epoch in 10 30 70 149 #160
do
    python main.py \
    --config "$conf_file" \
    --smart_ratio 0.9856 \
    --srV3-epoch $epoch \
    --subfolder "$subfolder_root$epoch" > "$log_root$epoch$log_end" 2>&1 &
done
BLOCK


