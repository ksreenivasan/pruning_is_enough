

# VGG16, 0.5%, bf_ft_acc vs af_ft_acc
:<<BLOCK
config_file="configs/hypercube/vgg16/vgg.yml"
subfolder_root="vgg_bf_af_relation_"
con="_"
log_end="_log"
sp_list=(0.5) #(0.5 1.4)
lr_list=(0.01 0.001 0.0001 0.00001)
#t_list=(100)  # 5 20 50 100
for sp in ${sp_list[@]}
do
    for lr in ${lr_list[@]}
    do
        python main.py \
        --config "$config_file" --subfolder "$subfolder_root$sp$con$lr" \
        --target-sparsity "$sp" --lr "$lr" > "$subfolder_root$sp$con$lr$log_end" 2>&1 &
    done
done
BLOCK



:<<BLOCK
# VGG16, bias=True, Affine-BN
config_file="configs/hypercube/vgg16/vgg_bias_affine.yml"
subfolder_root="vgg_bias_affine_True_hc_sparsity_"
log_end="_log"


sp_list=(50 5 2.5 1.4)
for sp in ${sp_list[@]}
do
    python main.py \
    --config "$config_file" --subfolder "$subfolder_root$sp" \
    --target-sparsity "$sp" #> "$subfloder_root$sp$log_end" 2>&1 &
done

BLOCK



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
    --subfolder "$subfolder_root" #> "$subfolder_root$log_end" 2>&1 &
BLOCK

# Renda 
#:<<BLOCK
conf_file="configs/imp/resnet18.yml"
subfolder_root="resnet18_cifar10_renda_"
log_end="_log"
gpu=1

python imp_main.py \
    --config "$conf_file" \
    --imp-rounds 20 \
    --imp-no-rewind \
    --gpu $gpu \
    --subfolder "$subfolder_root" > "$subfolder_root$log_end" 2>&1 &
#BLOCK



# EP
:<<BLOCK
conf_file="configs/ep/resnet18/resnet18_sc_ep.yml"
subfolder_root="resnet18_cifar10_ep_"
log_end="_log"

for pr in 0.05 0.02 0.005
do
    python main.py \
    --config "$conf_file" \
    --prune-rate "$pr" \
    --subfolder "$subfolder_root$pr" > "$subfolder_root$pr$log_end" 2>&1 &
done

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
    --subfolder "$subfolder_root$sr" > "$subfolder_root$sr$log_end" 2>&1 &
done
BLOCK

# Gem-Miner (hypercube)
:<<BLOCK

# 5% sparsity
#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_real_"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6_iter_20.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6_iter_20_multistep_0_1.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_multistep_0_1"

#conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6_iter_20_cosine_0_1.yml"
#subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_cosine_0_1"

conf_file="configs/hypercube/resnet18/cifar10/sparsity_5_85lam6_iter_20_adam.yml"
subfolder_root="resnet18_cifar10_hc_sparsity_5_iter_20_adam"

















# Running trials in parallel
:<<BLOCK
conf_file="configs/ablation_hc/resnet20_1_4_normal.yml"
log_root="hc_resnet20_1_4_"
log_end="_log"
subfolder_root="hc_resnet20_1_4_results_"

for trial in 2
do
    python main.py \
    --config "$conf_file" --subfolder "$subfolder_root$trial" \
    --trial-num "$trial" > "$log_root$trial$log_end" 2>&1 #&

#    python main.py \
#    --config "$conf_file" --subfolder "invert_$subfolder_root$trial" \
#    --trial-num "$trial" --invert-sanity-check --skip-sanity-checks > "invert_$log_root$trial$log_end" 2>&1 &
done
BLOCK


:<<BLOCK
conf_file="configs/ablation_hc/resnet20_1_4_unflag_False.yml"
log_root="hc_resnet20_1_4_unflag_False_"
log_end="_log"
subfolder_root="hc_resnet20_1_4_unflag_False_results_"

for trial in 1 2
do
    python main.py \
    --config "$conf_file" --subfolder "$subfolder_root$trial" \
    --trial-num "$trial" > "$log_root$trial$log_end" 2>&1 #&
done
BLOCK













### MobileNetV2
#####python main.py --config configs/training/mobilenetV2/cifar10_mobileV2_training.yml #> mobilenet_cifar10_wt



#python main.py --config configs/training/mobilenetV2/cifar10_mobileV2_training_check.yml #> mobilenet_cifar10_wt_check

:<<BLOCK
#python main.py --config configs/ep/mobilenetV2/cifar10_mobileV2_ep_sparsity_50.yml > log_mobilev2_EP_sparsity_50 2>&1
#python main.py --config configs/ep/mobilenetV2/cifar10_mobileV2_ep_sparsity_5.yml > log_mobilev2_EP_sparsity_5 2>&1
#python main.py --config configs/ep/mobilenetV2/cifar10_mobileV2_ep_sparsity_20.yml > log_mobilev2_EP_sparsity_20 2>&1
#python main.py --config configs/ep/mobilenetV2/cifar10_mobileV2_ep_sparsity_1_7.yml > log_mobilev2_EP_sparsity_1_7 2>&1
BLOCK

:<<BLOCK
python main.py --config configs/hypercube/mobilenetV2/sparsity_50.yml > log_mobilev2_HC_sparsity_50 2>&1 
python main.py --config configs/hypercube/mobilenetV2/sparsity_5.yml > log_mobilev2_HC_sparsity_5 2>&1 
python main.py --config configs/hypercube/mobilenetV2/sparsity_20.yml > log_mobilev2_HC_sparsity_20 2>&1 
BLOCK
#python main.py --config configs/hypercube/mobilenetV2/sparsity_5_7lam6.yml > log_mobilev2_HC_sparsity_5_7lam6 2>&1 
#python main.py --config configs/hypercube/mobilenetV2/sparsity_1_4.yml > log_mobilev2_HC_sparsity_1_4 2>&1 
#python main.py --config configs/hypercube/mobilenetV2/sparsity_20_3lam6.yml > log_mobilev2_HC_sparsity_20_3lam6 2>&1 

#python main.py --config configs/hypercube/mobilenetV2/sparsity_20_1lam6.yml > log_mobilev2_HC_sparsity_20_1lam6 2>&1 
#python main.py --config configs/hypercube/mobilenetV2/sparsity_20_3lam6.yml > log_mobilev2_HC_sparsity_20_3lam6 2>&1 


#### ResNet-18
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml # 93.17% at 150 epoch
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_v2.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml 
#python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_evaluate.yml 


### ResNet-32
#python main.py --config configs/training/resnet32/cifar10_resnet32_training.yml


### ResNet-20
#python main.py --config configs/ep/resnet20/resnet20_sc_ep.yml 
#TODO: add global_ep yml here
#python main.py --config configs/ep/resnet20/resnet20_global_ep_iter.yml #> cifar_log 2>&1
#python main.py --config configs/ep/resnet20/resnet20_global_ep_iter_adam.yml #> cifar_log 2>&1

# Smart Ratio
#python main.py --config configs/sr/resnet20/resnet20_sr.yml # ResNet-20
#python main.py --config configs/sr/resnet32/resnet32_sr.yml # ResNet-32
#python main.py --config configs/training/resnet20/cifar10_resnet20_training.yml
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t1.yml #> log_hc_sparsity_3_72_t1 2>&1


# check mixed precision
#python main.py --config configs/hypercube/resnet20/mixed_precision/resnet20_sparsity_3_72_t1_with_MP.yml #> log_hc_sparsity_3_72_t1 2>&1

# old pruning schedule
:<<BLOCK
python main.py --config configs/hypercube/resnet20/old_prune_schedule/resnet20_sparsity_1_8_t1.yml > log_hc_old_prune_schedule_sparsity_1_8_t1 2>&1
python main.py --config configs/hypercube/resnet20/old_prune_schedule/resnet20_sparsity_0_15_t1.yml > log_hc_old_prune_schedule_sparsity_0_15_t1 2>&1
python main.py --config configs/hypercube/resnet20/old_prune_schedule/resnet20_sparsity_0_15_t2.yml > log_hc_old_prune_schedule_sparsity_0_15_t2 2>&1
python main.py --config configs/hypercube/resnet20/old_prune_schedule/resnet20_sparsity_0_15_t3.yml > log_hc_old_prune_schedule_sparsity_0_15_t3 2>&1
python main.py --config configs/hypercube/resnet20/old_prune_schedule/resnet20_sparsity_0_15_t4.yml > log_hc_old_prune_schedule_sparsity_0_15_t4 2>&1
python main.py --config configs/hypercube/resnet20/old_prune_schedule/resnet20_sparsity_0_15_t5.yml > log_hc_old_prune_schedule_sparsity_0_15_t5 2>&1
BLOCK
#python main.py --config configs/hypercube/resnet20/old_prune_schedule/resnet20_target_sparsity_0_15_t1.yml > log_hc_old_prune_schedule_target_sparsity_0_15_t1 2>&1
#python main.py --config configs/hypercube/resnet20/old_prune_schedule/resnet20_target_sparsity_0_15_t1.yml > log_hc_old_prune_schedule_target_sparsity_0_15_t1_real 2>&1

# HC for multiple trials
#:<<BLOCK
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t2.yml > log_hc_sparsity_3_72_t2 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t3.yml > log_hc_sparsity_3_72_t3 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t4.yml > log_hc_sparsity_3_72_t4 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t5.yml > log_hc_sparsity_3_72_t5 2>&1
#BLOCK

:<<BLOCK
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t1.yml > log_hc_sparsity_0_59_t1 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t2.yml > log_hc_sparsity_0_59_t2 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t3.yml > log_hc_sparsity_0_59_t3 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t4.yml > log_hc_sparsity_0_59_t4 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t5.yml > log_hc_sparsity_0_59_t5 2>&1
BLOCK






# EP
#python main.py --config configs/ep/resnet20/resnet20_sc_ep_sparsity_50.yml #> log_EP_sparsity_50 2>&1
#python main.py --config configs/ep/resnet20/resnet20_sc_ep_sparsity_13_34.yml > log_EP_sparsity_13_34 2>&1
#python main.py --config configs/ep/resnet20/resnet20_sc_ep_sparsity_3_72.yml > log_EP_sparsity_3_72 2>&1
#python main.py --config configs/ep/resnet20/resnet20_sc_ep_sparsity_1_44.yml > log_EP_sparsity_1_44 2>&1
#python main.py --config configs/ep/resnet20/resnet20_sc_ep_sparsity_0_59.yml > log_EP_sparsity_0_59 2>&1
#python main.py --config configs/ep/resnet20/resnet20_sc_ep_sparsity_0_15.yml > log_EP_sparsity_0_15 2>&1
#python main.py --config configs/ep/resnet20/resnet20_sc_global_ep.yml 



## HC for denser models
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_5.yml > log_target_sparsity_5_lam_5e-6 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_20.yml > log_target_sparsity_20_lam_1e-6 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_50.yml > log_target_sparsity_50_lam_0 2>&1

#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_5.yml > log_target_sparsity_5_lam_3e-5 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_20.yml > log_target_sparsity_20_lam_1e-5 2>&1

#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_5_without_unflag.yml > log_target_sparsity_5_without_unflag_lam_3e-5 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_20_without_unflag.yml > log_target_sparsity_20_without_unflag_lam_1e-5 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_50_without_unflag.yml > log_target_sparsity_50_without_flag_lam_0 2>&1

#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_0_5_MAML_1.yml --run_idx 1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_0_5_MAML_1e-2.yml --run_idx 1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_0_5_MAML_1e-4.yml --run_idx 2
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_0_5_MAML_0.yml --run_idx 3




#python main.py --config config10.yml --run_idx 10 #> log_config$r 2>&1	
:<<BLOCK
run_list=(3 4)
for r in ${run_list[@]}
do
	python main.py --config config$r.yml --run_idx $r #> log_config$r 2>&1	
done
BLOCK

#python main.py --config config1.yml --run_idx 1
#python main.py --config config2.yml --run_idx 2 
#python main.py --config config3.yml --run_idx 3
#python main.py --config config4.yml --run_idx 4 

#python main.py --config config5.yml --run_idx 5
#python main.py --config config6.yml --run_idx 6
#python main.py --config config7.yml --run_idx 7
#python main.py --config config8.yml --run_idx 8 

#python main.py --config config9.yml --run_idx 9
#python main.py --config config10.yml --run_idx 10

#python main.py --config config11.yml --run_idx 11
#python main.py --config config12.yml --run_idx 12
#python main.py --config config13.yml --run_idx 13




:<<BLOCK
run_list=(2 3)
for r in ${run_list[@]}
do
	python main.py --config config$r.yml --run_idx $r #> log_config$r 2>&1	
done
BLOCK

# To run: nohup bash cifar_exec_GD.sh &
# To view log: tail -f log_config_i



