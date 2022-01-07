# TinyImageNet, ResNet-18

# Weight training
#python main.py --config configs/training/resnet18/tiny_resnet18_training_preproc2_v3.yml #> log_tiny_wt_p2_v3 2>&1 # this is current best: 49.59%

# HC
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_sgd_5.yml  > log_tiny_hc_sparsity_50_sgd_5 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_sgd_25.yml  > log_tiny_hc_sparsity_50_sgd_25 2>&1

#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_adam_5.yml  > log_tiny_hc_sparsity_50_adam_5 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_adam_25.yml  > log_tiny_hc_sparsity_50_adam_25 2>&1


#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_sgd_lam4.yml  > log_tiny_hc_sparsity_5_sgd_lam4 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_sgd_lam5.yml  > log_tiny_hc_sparsity_5_sgd_lam5 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_sgd_lam6.yml  > log_tiny_hc_sparsity_5_sgd_lam6 2>&1


python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_adam_lam4.yml  > log_tiny_hc_sparsity_5_adam_lam4 2>&1
python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_adam_lam5.yml  > log_tiny_hc_sparsity_5_adam_lam5 2>&1
python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_adam_lam6.yml  > log_tiny_hc_sparsity_5_adam_lam6 2>&1



# EP
:<<BLOCK
python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_100.yml > log_tiny_ep_sparsity_100 2>&1
python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_50.yml > log_tiny_ep_sparsity_50 2>&1
python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_20.yml > log_tiny_ep_sparsity_20 2>&1
python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_5.yml > log_tiny_ep_sparsity_5 2>&1
python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_1_4.yml > log_tiny_ep_sparsity_1_4 2>&1
BLOCK


# testing mixed precision
#python main.py --config configs/training/resnet18/tiny_resnet18_training_test_MP.yml #> log_tiny_wt_p2_v3 2>&1
