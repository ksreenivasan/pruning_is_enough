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

#:<<BLOCK
# Using validation to figure out hyperparams
# NOTE: make sure to delete/comment subfolder from the config file or else it may not work
conf_file="configs/warm_gm/resnet20/resnet20_sp1_44_warm_gm"
conf_end=".yml"
log_root="resnet20_warm_gm_from_epoch_"
log_end="_log"
subfolder_root="resnet20_warm_gm_from_epoch_"
ckpt_path="model_checkpoints/resnet20_wt/wt_model_after_epoch_"

for epoch in 5 10
do
    python main.py \
    --config "$conf_file$conf_end" \
    --pretrained "${ckpt_path}${epoch}.pth" \
    --gpu 1 \
    --use-full-data \
    --subfolder "$subfolder_root$epoch" > "$log_root$epoch$log_end" 2>&1 &

    #python main.py \
    #--config "$conf_file$conf_end" \
    #--trial-num $trial \
    #--invert-sanity-check \
    #--skip-sanity-checks \
    #--subfolder "invert_$subfolder_root$trial" > "invert_$log_root$trial$log_end" 2>&1 &
done
#BLOCK

:<<BLOCK
# Final run on full data
# NOTE: make sure to delete/comment subfolder from the config file or else it may not work
conf_file="configs/param_tuning/resnet20_13_34/conf2"
conf_end=".yml"
log_root="resnet20_sp13_34_"
log_end="_log"
subfolder_root="resnet20_sp13_34_"

for trial in 1
do
    python main.py \
    --config "$conf_file$conf_end" \
    --trial-num $trial \
    --use-full-data \
    --subfolder "$subfolder_root$trial" > "$log_root$trial$log_end" 2>&1 &

    python main.py \
    --config "$conf_file$conf_end" \
    --trial-num $trial \
    --invert-sanity-check \
    --use-full-data \
    --skip-sanity-checks \
    --subfolder "invert_$subfolder_root$trial" > "invert_$log_root$trial$log_end" 2>&1 &
done
BLOCK
