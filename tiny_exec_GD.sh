# TinyImageNet, ResNet-18

# Weight training
#python main.py --config configs/training/resnet18/tiny_resnet18_training.yml

# HC
python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_t1.yml  #> log_tiny_hc_sparsity_50_t1 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_1_44_t1.yml  #> log_tiny_hc_sparsity_1_44_t1 2>&1

# EP
#python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_50.yml #> log_tiny_ep_sparsity_50 2>&1

