# Invert

## Local
#python main.py --config configs/hypercube/mobilenetV2/sparsity_20_3lam6_invert.yml > log_mobilev2_HC_sparsity_20_3lam6_invert 2>&1 
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_sgd_lam0_unflag_F_invert.yml > log_tiny_resnet18_sparsity_50_invert 2>&1


## AWS1
#python main.py --config configs/hypercube/mobilenetV2/sparsity_5_7lam6_unflag_True_invert.yml > log_mobilev2_HC_sparsity_5_7lam6_unflag_True_invert 2>&1 
#python main.py --config configs/hypercube/mobilenetV2/sparsity_1_4_invert.yml > log_mobilev2_HC_sparsity_1_4_invert 2>&1
#python main.py --config configs/hypercube/mobilenetV2/sparsity_50_invert.yml > log_mobilev2_HC_sparsity_50_invert 2>&1
#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_5_adam_8lam6_invert.yml > log_tiny_resnet18_sparsity_5_invert 2>&1




# Others (weight reinit, mask shuffle)
#python main.py --config configs/sanity/mobilenet_sanity.yml > log_sanity_mobilenet_sparsity_20 2>&1
python main.py --config configs/sanity/resnet18_sanity.yml > log_sanity_resnet18_sparsity_50 2>&1




