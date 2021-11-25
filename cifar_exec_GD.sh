#export cuda_visible_devices=3

#### ResNet-18
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml # 93.17% at 150 epoch
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_v2.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml 
#python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_evaluate.yml 


### ResNet-20
#python main.py --config configs/ep/resnet20/resnet20_sc_ep.yml 
python main.py --config configs/ep/resnet20/resnet20_sc_global_ep.yml 

:<<BLOCK
run_list=(3 4)
for r in ${run_list[@]}
do
	python main.py --config config$r.yml --run_idx $r #> log_config$r 2>&1	
done
BLOCK

:<<BLOCK
run_list=(5 6)
for r in ${run_list[@]}
do
	python main.py --config config$r.yml --run_idx $r #> log_config$r 2>&1	
done
BLOCK

:<<BLOCK
run_list=(7 8)
for r in ${run_list[@]}
do
	python main.py --config config$r.yml --run_idx $r #> log_config$r 2>&1	
done
BLOCK

:<<BLOCK
run_list=(1)
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


