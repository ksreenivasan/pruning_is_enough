# ResNet50_tf (changed the output size)

# Weight training
#python main.py --config configs/training/resnet50/caltech_resnet50_training_1FC.yml #> log_caltech_wt_50epoch_1FC 2>&1
#python main.py --config configs/training/resnet50/caltech_resnet50_training_2FC.yml #> log_caltech_wt_50epoch_2FC 2>&1



# HC
python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_50_1FC.yml #> log_caltech_hc_sparsity_50_1FC 2>&1
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_50_2FC.yml #> log_caltech_hc_sparsity_50_2FC 2>&1


#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_5_1FC.yml #> log_caltech_hc_sparsity_5_1FC 2>&1
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_5_2FC.yml #> log_caltech_hc_sparsity_5_2FC 2>&1


#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_2_1FC.yml #> log_caltech_hc_sparsity_2_1FC 2>&1
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_2_2FC.yml #> log_caltech_hc_sparsity_2_2FC 2>&1








# ======== OLD







# HC
#:<<BLOCK
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_50.yml > log_caltech_hc_sparsity_50_UV_1000_bias_real_final 2>&1
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_unconstrained.yml > log_caltech_hc_sparsity_unconstrained_UV_1000_bias_real_final 2>&1
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_5.yml > log_caltech_hc_sparsity_5_2lam6_UV 2>&1
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_2.yml > log_caltech_hc_sparsity_2_5lam6_UV 2>&1
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_0_5.yml > log_caltech_hc_sparsity_0_5_1_5lam5_UV 2>&1
#BLOCK
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_0_5.yml > log_caltech_hc_sparsity_0_5_2lam5_50epoch_mutli 2>&1
#python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_0_5_1lam5.yml > log_caltech_hc_sparsity_0_5_1lam5_50epoch_mutli 2>&1

# EP
#:<<BLOCK
#python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_50.yml > log_caltech_ep_sparsity_50_UV_1000_bias_real_final 2>&1
#python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_5.yml > log_caltech_ep_sparsity_5_UV 2>&1
#python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_2.yml > log_caltech_ep_sparsity_2_UV 2>&1
#python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_0_5.yml > log_caltech_ep_sparsity_0_5_UV 2>&1
#BLOCK

