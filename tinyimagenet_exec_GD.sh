


# TinyImageNet, ResNet-101

# Weight training
python main.py --config configs/training/resnet101/tiny_resnet101_training.yml #> log_tiny_res101_wt 2>&1 # this is current best







# TinyImageNet, ResNet-18


# Weight training
#python main.py --config configs/training/resnet18/tiny_resnet18_training_preproc2_v3.yml #> log_tiny_wt_p2_v3 2>&1 # this is current best: 49.59%

# HC
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_sgd_5.yml  > log_tiny_hc_sparsity_50_sgd_5 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_sgd_25.yml  > log_tiny_hc_sparsity_50_sgd_25 2>&1

#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_adam_5.yml  > log_tiny_hc_sparsity_50_adam_5 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_adam_25.yml  > log_tiny_hc_sparsity_50_adam_25 2>&1


#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_sgd_lam8.yml  #> log_tiny_hc_sparsity_5_sgd_lam8 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_sgd_lam8_T10.yml  #> log_tiny_hc_sparsity_5_sgd_lam8 2>&1

#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_sgd_lam0_T10.yml  > log_tiny_hc_sparsity_50_sgd_lam0_T10 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_20_sgd_lam8_T10.yml  > log_tiny_hc_sparsity_20_sgd_lam8_T10 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_1_4_sgd_lam7_T10.yml  > log_tiny_hc_sparsity_1_4_sgd_lam7_T10 2>&1

#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_sgd_lam0_unflag_F.yml > log_tiny_hc_sparsity_50_sgd_lam0_unflag_F 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_20_sgd_lam6_unflag_F.yml > log_tiny_hc_sparsity_20_sgd_lam6_unflag_F 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_20_sgd_lam5_unflag_F.yml > log_tiny_hc_sparsity_20_sgd_lam5_unflag_F 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_sgd_lam4_unflag_F.yml > log_tiny_hc_sparsity_5_sgd_lam4_unflag_F 2>&1

#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_sgd_5lam6_unflag_F.yml > log_tiny_hc_sparsity_5_sgd_5lam6_unflag_F 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_20_sgd_3lam6_unflag_F.yml > log_tiny_hc_sparsity_20_sgd_3lam6_unflag_F 2>&1

#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_adam_8lam6.yml > log_tiny_hc_sparsity_5_adam_8lam6 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_1_4_adam_9lam6.yml #> log_tiny_hc_sparsity_1_4_adam_9lam6 2>&1

#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_adam_1lam6.yml > log_tiny_hc_sparsity_5_adam_1lam6 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_1_4_adam_5lam6.yml > log_tiny_hc_sparsity_1_4_adam_5lam6 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_0_5_adam_1lam5.yml > log_tiny_hc_sparsity_0_5_adam_1lam5 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_0_5_sgd_1_5lam5.yml > log_tiny_hc_sparsity_0_5_sgd_1_5lam5 2>&1



# EP
#:<<BLOCK
#python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_5.yml > log_tiny_ep_sparsity_5 2>&1
#python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_50.yml > log_tiny_ep_sparsity_50 2>&1
#python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_100.yml > log_tiny_ep_sparsity_100 2>&1
#python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_1_4.yml > log_tiny_ep_sparsity_1_4 2>&1
#python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_20.yml > log_tiny_ep_sparsity_20 2>&1
#BLOCK
#python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_0_75.yml > log_tiny_ep_sparsity_0_75 2>&1


# testing mixed precision
#python main.py --config configs/training/resnet18/tiny_resnet18_training_test_MP.yml #> log_tiny_wt_p2_v3 2>&1
