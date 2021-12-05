
### ResNet-20
init_list=(unif)
#init_list=(skew unif)
for init in ${init_list[@]}
do
#	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov20_v2.yml \
#		--iter_period 25 --prune-rate 0.55 --score-init $init --gpu 1 --lmbda 0.0001
#	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov20_v2.yml \
#		--iter_period 25 --prune-rate 0.55 --score-init $init --gpu 2 --lmbda 0.0001 --fine-tune-lr 0.1
#	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov20_v2.yml \
#		--iter_period 25 --prune-rate 0.55 --score-init $init --gpu 2 --lmbda 0 --fine-tune-lr 0.1
	python main.py --config configs/hypercube/resnet20/resnet20_hypercube_bottom_K_Nov22_3pm.yml \
		--iter_period 5 --prune-rate 0.5 --score-init $init --gpu 2 --lmbda 0.00005 --fine-tune-lr 0.01
done