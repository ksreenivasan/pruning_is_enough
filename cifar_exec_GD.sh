#export cuda_visible_devices=3

### MobileNetV2
#####python main.py --config configs/training/mobilenetV2/cifar10_mobileV2_training.yml #> mobilenet_cifar10_wt



#python main.py --config configs/training/mobilenetV2/cifar10_mobileV2_training_check.yml #> mobilenet_cifar10_wt_check

#:<<BLOCK
#python main.py --config configs/ep/mobilenetV2/cifar10_mobileV2_ep_sparsity_50.yml > log_mobilev2_EP_sparsity_50 2>&1
#python main.py --config configs/ep/mobilenetV2/cifar10_mobileV2_ep_sparsity_5.yml > log_mobilev2_EP_sparsity_5 2>&1
#python main.py --config configs/ep/mobilenetV2/cifar10_mobileV2_ep_sparsity_20.yml > log_mobilev2_EP_sparsity_20 2>&1
#python main.py --config configs/ep/mobilenetV2/cifar10_mobileV2_ep_sparsity_1_7.yml > log_mobilev2_EP_sparsity_1_7 2>&1


#BLOCK

:<<BLOCK
python main.py --config configs/hypercube/mobilenetV2/sparsity_50.yml > log_mobilev2_HC_sparsity_50 2>&1 
python main.py --config configs/hypercube/mobilenetV2/sparsity_5.yml > log_mobilev2_HC_sparsity_5 2>&1 
python main.py --config configs/hypercube/mobilenetV2/sparsity_20.yml > log_mobilev2_HC_sparsity_20 2>&1 
BLOCK
#python main.py --config configs/hypercube/mobilenetV2/sparsity_5_7lam6.yml > log_mobilev2_HC_sparsity_5_7lam6 2>&1 
#python main.py --config configs/hypercube/mobilenetV2/sparsity_20_1lam6.yml > log_mobilev2_HC_sparsity_20_1lam6 2>&1 


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
#python main.py --config configs/training/resnet20/cifar10_resnet20_training.yml
#python main.py --config configs/hypercube/resnet20/resnet20_sparsity_3_72_unflagT.yml > log_hc_sparsity_3_72_unflagT 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_sparsity_1_44_unflagT.yml > log_hc_sparsity_1_44_unflagT 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_sparsity_0_59_unflagT.yml > log_hc_sparsity_0_59_unflagT 2>&1





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

#python main.py --config config6.yml --run_idx 6
#python main.py --config config5.yml --run_idx 5
#python main.py --config config8.yml --run_idx 8 
#python main.py --config config7.yml --run_idx 7

#python main.py --config config9.yml --run_idx 9
#python main.py --config config10.yml --run_idx 10

#python main.py --config config11.yml --run_idx 11

:<<BLOCK
run_list=(2 3)
for r in ${run_list[@]}
do
	python main.py --config config$r.yml --run_idx $r #> log_config$r 2>&1	
done
BLOCK

# To run: nohup bash cifar_exec_GD.sh &
# To view log: tail -f log_config_i



