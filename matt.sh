#for period in 15
#do
#       python main.py \
#                   --config configs/hypercube/resnet20/finetune_check/finetune_lam_1_num_5_without_unflag.yml \
#                   --iter-period $period \
#                   --gpu 2
#done

#for l in 0.000001  # 5 2.5 1.4 0.5
#do
#       python main.py \
#	      	--config configs/hypercube/vgg16/vgg.yml \
#                   --lmbda 0.000001 \
#                   --gpu 0 \
#                   --target-sparsity 50 \
#                   --subfolder VGG_HC_unflag
#done

#python main.py --config configs/hypercube/vgg16/vgg16.yml
#for p in .014 .005 
#do 
#	python main.py --config configs/ep/vgg/vgg16_sc_ep.yml \
#		--prune-rate $p		
#done

for l in 0.05 0.01
do
	python main.py --config configs/training/vgg16/vgg_WT.yml \
		--lr $l
done
