#export cuda_visible_devices=3

# ResNet-18

#### ResNet-18
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml # 93.17% at 150 epoch
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_v2.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml 
#python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_evaluate.yml 




# ResNet-20
#python main.py --config configs/ep/resnet20/resnet20_sc_ep.yml #> log_EP_sparsity_50 2>&1

## HC for multiple trials
:<<BLOCK
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t2.yml > log_hc_sparsity_3_72_t2 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t3.yml > log_hc_sparsity_3_72_t3 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t4.yml > log_hc_sparsity_3_72_t4 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t5.yml > log_hc_sparsity_3_72_t5 2>&1
BLOCK

:<<BLOCK
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t2.yml > log_hc_sparsity_0_59_t2 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t3.yml > log_hc_sparsity_0_59_t3 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t4.yml > log_hc_sparsity_0_59_t4 2>&1
python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t5.yml > log_hc_sparsity_0_59_t5 2>&1
BLOCK

## ran in finetune_loss repo
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_1_35_t1.yml > log_hc_sparsity_1_35_t1 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_1_35_t2.yml > log_hc_sparsity_1_35_t2 2>&1

#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_1_35_t3.yml > log_hc_sparsity_1_35_t3 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_1_35_t4.yml > log_hc_sparsity_1_35_t4 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_1_35_t5.yml > log_hc_sparsity_1_35_t5 2>&1



## EP
#python main.py --config configs/ep/resnet20/resnet20_sc_ep_sparsity_50.yml > log_EP_sparsity_50 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t1.yml #> log_hc_sparsity_3_72_t1 2>&1

# HC for multiple trials
#:<<BLOCK
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t1.yml > log_hc_sparsity_3_72_t1 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t2.yml > log_hc_sparsity_3_72_t2 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t3.yml > log_hc_sparsity_3_72_t3 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t4.yml > log_hc_sparsity_3_72_t4 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_3_72_t5.yml > log_hc_sparsity_3_72_t5 2>&1
#BLOCK

#:<<BLOCK
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t1.yml > log_hc_sparsity_0_59_t1 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t2.yml > log_hc_sparsity_0_59_t2 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t3.yml > log_hc_sparsity_0_59_t3 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t4.yml > log_hc_sparsity_0_59_t4 2>&1
#python main.py --config configs/hypercube/resnet20/error_bar/resnet20_sparsity_0_59_t5.yml > log_hc_sparsity_0_59_t5 2>&1
#BLOCK



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


## adding finetune for denser models
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_5_without_unflag_with_finetune.yml > log_target_sparsity_5_without_unflag_with_finetune_lam_3e-5 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_20_without_unflag_with_finetune.yml > log_target_sparsity_20_without_unflag_with_finetune_lam_1e-5 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_50_without_unflag_with_finetune.yml > log_target_sparsity_50_without_unflag_with_finetune_lam_0 2>&1

#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_5_without_unflag_with_finetune_last_10_epochs.yml > log_target_sparsity_5_without_unflag_with_finetune_last_10_epochs_lam_3e-5 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_20_without_unflag_with_finetune_last_10_epochs.yml > log_target_sparsity_20_without_unflag_with_finetune_last_10_epochs_lam_1e-5 2>&1
#python main.py --config configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_50_without_unflag_with_finetune_last_10_epochs.yml > log_target_sparsity_50_without_unflag_with_finetune_last_10_epochs_lam_0 2>&1


## testing adding finetune loss
#python main.py --config configs/hypercube/resnet20/finetune_check/no_finetune.yml > log_no_finetune_5e-5 2>&1
#python main.py --config configs/hypercube/resnet20/finetune_check/finetune_lam_1_num_5_last_10_epochs_without_unflag.yml > log_finetune_1e-4_lam_1_num_5_last_10_epochs_without_unflag 2>&1
#python main.py --config configs/hypercube/resnet20/finetune_check/finetune_lam_1_num_5_last_10_epochs.yml > log_finetune_1e-4_lam_1_num_5_last_10_epochs 2>&1
#python main.py --config configs/hypercube/resnet20/finetune_check/finetune_lam_1_num_5_last_10_epochs_without_unflag.yml > log_finetune_5e-5_lam_1_num_5_last_10_epochs_without_unflag 2>&1
#python main.py --config configs/hypercube/resnet20/finetune_check/finetune_lam_1_num_5_last_10_epochs.yml > log_finetune_5e-5_lam_1_num_5_last_10_epochs 2>&1


#python main.py --config configs/hypercube/resnet20/finetune_check/no_finetune_without_unflag.yml
#python main.py --config configs/hypercube/resnet20/finetune_check/finetune_lam_1_num_5_without_unflag.yml
#python main.py --config configs/hypercube/resnet20/finetune_check/finetune_lam_0_1_num_5_without_unflag.yml



#python main.py --config config1.yml --run_idx 1 #  
#python main.py --config config1.yml --run_idx 0 #  
#python main.py --config config2.yml --run_idx 2
#python main.py --config configs/hypercube/resnet20/finetune_check/finetune_lam_1_num_10.yml
#python main.py --config configs/hypercube/resnet20/finetune_check/finetune_lam_0_1_num_10.yml









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










#	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov22_8pm_SGD.yml \
#		--iter_period 8 --prune-rate 0.2 --score-init $init --gpu 3 --lmbda 0 --fine-tune-lr 0.1
#	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov22_8pm_SGD.yml \
#		--iter_period 8 --prune-rate 0.2 --score-init $init --gpu 3 --lmbda 0.000001 --fine-tune-lr 0.1
#	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov22_10pm_SGD.yml \
#		--iter_period 8 --prune-rate 0.2 --score-init $init --gpu 3 --lmbda 0.00005 --fine-tune-lr 0.01 \
#		--regularization L1 --lr-policy constant_lr  
#	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov22_10pm_SGD.yml \
#		--iter_period 8 --prune-rate 0.2 --score-init $init --gpu 3 --lmbda 0.00005 --fine-tune-lr 0.01 \
#		--regularization L1 --lr-policy cosine_lr 
#	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov22_10pm_SGD.yml \
#		--iter_period 8 --prune-rate 0.2 --score-init $init --gpu 3 --lmbda 0.00005 --fine-tune-lr 0.01 \
#		--regularization L1 --lr-policy cosine_lr 
#	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov23_11am_SGD.yml \
#		--iter_period 49 --prune-rate 0.98 --score-init $init --gpu 3 --lmbda 0.0001 --fine-tune-lr 0.01 \
#		--regularization L1 --lr-policy cosine_lr


:<<BLOCK
sp_list=(1 2 3 4 5 6 8 10 13 16 20 26 32 40 51 63 79)

for sp in ${sp_list[@]}
do
	python main.py \
   	--config configs/hypercube/resnet20/resnet20_sc_hypercube_reg_bottom_K_periodic_rounding.yml \
   	--pretrained model_checkpoints/resnet20/hc_ckpt_at_sparsity_$sp.pt \
	--shuffle \
	--chg_mask \
   	--results-filename results/finetune_shuffle_mask_$sp.csv
done
#python main.py --config configs/hypercube/resnet20/resnet20_wt.yml
BLOCK

:<<BLOCK
pr_list=(8)
#pr_list=(7 8 9 10 11 13 15 19 25 38 75)


for pr in ${pr_list[@]}
do
	python main.py --config configs/hypercube/resnet20/resnet20_quantized_hypercube_reg_bottom_K_GD.yml \
		--iter_period $pr \
		--skip-sanity-checks
done
BLOCK



#python main.py --config configs/hypercube/resnet20/resnet20_quantized_hypercube_reg_bottom_K_tinyImageNet.yml


