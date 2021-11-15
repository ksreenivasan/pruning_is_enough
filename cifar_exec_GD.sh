#export cuda_visible_devices=3

#### ResNet-18
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml # 93.17% at 150 epoch
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_v2.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml 
#python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_evaluate.yml 


### ResNet-20
:<<BLOCK
sp_list=(1 2 3 4 5 6 8 10 13 16 20 26 32 40 51 63 79)

for sp in ${sp_list[@]}
do
	python main.py \
   	--config configs/hypercube/resnet20/resnet20_sc_hypercube_reg_bottom_K_periodic_rounding.yml \
   	--random-subnet \
   	--pretrained model_checkpoints/resnet20/hc_ckpt_at_sparsity_$sp.pt \
	--shuffle \
	--chg_mask \
   	--results-filename results/finetune_shuffle_mask_$sp.csv
done
#python main.py --config configs/hypercube/resnet20/resnet20_wt.yml
BLOCK

:<<BLOCK
pr_list=(7 8 9 10 11 13 15 19 25 38 75)

for pr in ${pr_list[@]}
do
	python main.py --config configs/hypercube/resnet20/resnet20_quantized_hypercube_reg_bottom_K.yml \
		--iter_period $pr
done
BLOCK


#:<<BLOCK
rate_list=(0.01 0.02 0.03 0.05 0.08 0.13 0.20 0.32 0.51 0.80)

for rate in ${rate_list[@]}
do
	python main.py --config configs/ep/resnet20/resnet20_sc_ep.yml \
		--prune-rate $rate \
        	--skip-fine-tune
done
#BLOCK




