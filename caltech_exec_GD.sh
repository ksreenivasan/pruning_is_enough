# ResNet50_tf (changed the output size)

# Weight training
python main.py --config configs/training/resnet50/caltech_resnet50_training.yml

# HC
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_1_4_adam_9lam6.yml > log_tiny_hc_sparsity_1_4_adam_9lam6 2>&1

# EP
#python main.py --config configs/ep/tinyImageNet/resnet18/sparsity_0_75.yml > log_tiny_ep_sparsity_0_75 2>&1


