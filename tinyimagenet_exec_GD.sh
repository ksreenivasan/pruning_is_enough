# TinyImageNet, ResNet-18

# Weight training
## data preprocessing optoin #1
#python main.py --config configs/training/resnet18/tiny_resnet18_training.yml > log_tiny_wt 2>&1
#python main.py --config configs/training/resnet18/tiny_resnet18_training_v2.yml > log_tiny_wt_v2 2>&1
#python main.py --config configs/training/resnet18/tiny_resnet18_training_v3.yml > log_tiny_wt_v3 2>&1
#python main.py --config configs/training/resnet18/tiny_resnet18_training_v3_bias_on.yml > log_tiny_wt_v3_bias_on 2>&1
## data preprocessing option #2
python main.py --config configs/training/resnet18/tiny_resnet18_training_preproc2_v3.yml #> log_tiny_wt_p2_v3 2>&1 # this is current best: 49.59%
#python main.py --config configs/training/resnet18/tiny_resnet18_training_preproc2_v3_bias_on.yml > log_tiny_wt_p2_v3_bias_on 2>&1

# HC
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_t1.yml  > log_tiny_hc_sparsity_50_t1 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_1_4_t1.yml  > log_tiny_hc_sparsity_1_4_t1 2>&1

# EP
#python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_50.yml #> log_tiny_ep_sparsity_50 2>&1



# testing mixed precision
#python main.py --config configs/training/resnet18/tiny_resnet18_training_test_MP.yml #> log_tiny_wt_p2_v3 2>&1
