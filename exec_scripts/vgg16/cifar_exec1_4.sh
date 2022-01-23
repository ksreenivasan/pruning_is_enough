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
# NOTE: make sure to delete/comment subfolder from the config file or else it may not work
conf_file="configs/hypercube/vgg16/vgg.yml"
log_root="vgg16_sp1_4_"
log_end="_log"
subfolder_root="vgg16_sp1_4_results_trial_"

for trial in 3
do
    python main.py \
    --config "$conf_file" \
    --trial-num $trial \
    --target-sparsity 1.4 \
    --gpu 0 \
    --subfolder "$subfolder_root$trial" > "$log_root$trial$log_end" 2>&1 &

    python main.py \
    --config "$conf_file" \
    --trial-num $trial \
    --invert-sanity-check \
    --target-sparsity 1.4 \
    --gpu 0 \
    --subfolder "invert_$subfolder_root$trial" > "invert_$log_root$trial$log_end" 2>&1 &
done
