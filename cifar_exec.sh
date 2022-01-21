#### ResNet-18
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml # 93.17% at 150 epoch
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg.yml 
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_v2.yml 
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml 
# python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_evaluate.yml 


### ResNet-20
:<<BLOCK
# python main.py \
#    --config configs/hypercube/resnet20/resnet20_random_subnet.yaml > kartik_log 2>&1
BLOCK

:<<BLOCK
python main.py --config configs/hypercube/resnet20/resnet20_sc_hypercube_reg_GD.yml
BLOCK

# HC
:<<BLOCK
python main.py \
--config configs/hypercube/resnet20/resnet20_quantized_iter_hc_0_75.yml > cifar_log 2>&1
BLOCK

# target sparsity 0.5
:<<BLOCK
python main.py \
--config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_0_5.yml > cifar_log_target_0_5 2>&1
BLOCK

# EP
:<<BLOCK
python main.py \
--config cibfugs/ep/resnet20/resnet20_global_ep.yml > cifar_log 2>&1
BLOCK

# add score rewinding
# python main.py \
#    --config configs/hypercube/resnet20/resnet20_quantized_hypercube_reg_bottom_K_rewind.yml


# python main.py --config configs/hypercube/resnet20/resnet20_wt.yml

:<<BLOCK
python main.py \
    --config configs/ep/resnet18/resnet18_sc_ep.yml > cifar_log 2>&1
BLOCK

#### Conv4

# Weight training (WT)
#python main.py --config configs/training/conv4/conv4_training.yml

# EP
#python main.py --config configs/ep/conv4/conv4_sc_ep.yml 

## WideResNet28
:<<BLOCK
python main.py \
--config configs/hypercube/wideresnet28/wideresnet28_weight_training.yml > wideresnet_wt_log 2>&1
BLOCK


# Running trials in parallel
conf_file="configs/hypercube/wideresnet28/wideresnet28_weight_training.yml"
log_root="wideresnet28_"
log_end="_log"
subfolder_root="wideresnet28_results_"

for trial in 2 3
do
    python main.py \
    --config "$conf_file" > "$log_root$trial$log_end" \
    --subfolder "$subfolder_root$trial" 2>&1 &
done