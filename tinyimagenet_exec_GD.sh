

# TinyImageNet, MobilenetV2

# Weight training (WT)
#python main.py --config configs/training/mobilenetV2/tiny_adam.yml #> log_tiny_mobile_wt_adam_0001_multi 2>&1 
#python main.py --config configs/training/mobilenetV2/tiny_sgd.yml > log_tiny_mobile_wt_sgd_01_multi 2>&1 

# Renda
#:<<BLOCK
gpu=1
subfolder="tiny_mobile_renda_updated"
python imp_main.py --config configs/imp/tiny_mobilenet.yml --imp-rounds 20 --imp-no-rewind --gpu $gpu --subfolder "$subfolder" > "$subfolder" 2>&1
#BLOCK

# IMP
:<<BLOCK
gpu=1
subfolder="tiny_mobile_imp"
python imp_main.py --config configs/imp/tiny_mobilenet.yml --imp-rounds 20 --gpu $gpu --subfolder "$subfolder" > "$subfolder" 2>&1
BLOCK

# smart ratio (SR)
####### go to SR after getting the best result for WT
:<<BLOCK
gpu=0
subfolder="tiny_mobile_sr_20_cosien"
python main.py --config configs/sr/tiny_sr_mobilenet.yml --gpu $gpu --smart_ratio 0.8 --subfolder "$subfolder" > "$subfolder" 2>&1
#python main.py --config configs/sr/tiny_sr_mobilenet.yml --gpu $gpu --smart_ratio 0.8 --subfolder "$subfolder" > "$subfolder" 2>&1
BLOCK

# Gem-Miner (GM)
:<<BLOCK
gpu=1
sp=20
lmbda=0.000003 #(0.00008 0.00003)
subfolder="tiny_mobile_gm_sp_20_lam_3e6_sgd_unflag_F"

python main.py --config configs/hypercube/tinyImageNet/mobilenetV2/sgd_unflag_F.yml \
			--gpu $gpu --target-sparsity $sp --lmbda $lmbda --subfolder "$subfolder" > "$subfolder" 2>&1
BLOCK



# Gem-Miner (GM) - Sanity checks!
:<<BLOCK
gpu=0
subfolder="tiny_mobile_gm_sp_20_sanity"
python main.py --config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_20_sgd_w_sanity.yml \
			--gpu $gpu --subfolder "$subfolder" > "$subfolder" 2>&1

gpu=0
subfolder="tiny_mobile_gm_sp_3_6_sanity"
python main.py --config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_3_6_adam_w_sanity.yml \
			--gpu $gpu --subfolder "$subfolder" > "$subfolder" 2>&1

gpu=1
subfolder="tiny_mobile_gm_sp_1_4_sanity"
python main.py --config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_1_4_sgd_w_sanity.yml \
			--gpu $gpu --subfolder "$subfolder" > "$subfolder" 2>&1

BLOCK

gpu=3
subfolder="tiny_mobile_gm_sp_20_invert"
python main.py --config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_20_sgd_w_invert.yml \
			--gpu $gpu --subfolder "$subfolder" > "$subfolder" 2>&1

subfolder="tiny_mobile_gm_sp_3_6_invert"
python main.py --config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_3_6_adam_w_invert.yml \
			--gpu $gpu --subfolder "$subfolder" > "$subfolder" 2>&1

subfolder="tiny_mobile_gm_sp_1_4_invert"
python main.py --config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_1_4_sgd_w_invert.yml \
			--gpu $gpu --subfolder "$subfolder" > "$subfolder" 2>&1

#:<<BLOCK
#BLOCK


# Edge-Popup (EP)
:<<BLOCK
gpu=1
subfolder="tiny_mobile_ep_sp_5_sgd" 
python main.py --config configs/ep/tinyImageNet/mobilenetV2/sparsity_5.yml \
			--gpu $gpu --subfolder "$subfolder"  > "$subfolder" 2>&1
BLOCK

:<<BLOCK
gpu=1
subfolder="tiny_mobile_ep_sp_20_sgd"
python main.py --config configs/ep/tinyImageNet/mobilenetV2/sparsity_20.yml \
			--gpu $gpu --subfolder "$subfolder"  > "$subfolder" 2>&1
BLOCK

:<<BLOCK
gpu=0
subfolder="tiny_mobile_ep_sp_1_4_sgd"
python main.py --config configs/ep/tinyImageNet/mobilenetV2/sparsity_1_4.yml \
			--gpu $gpu --subfolder "$subfolder"  > "$subfolder" 2>&1
BLOCK



#python main.py --config configs/hypercube/resnet18/resnet18_sparsity_1_4_adam_5lam6.yml # resnet18, check code




# TinyImageNet, ResNet-50

# Weight training
#python main.py --config configs/training/resnet50/tiny_resnet50_training_adam_001_multi.yml > log_tiny_res50_wt_adam_001_multi 2>&1 # this is current best
#python main.py --config configs/training/resnet50/tiny_resnet50_training_adam_001_cosine.yml > log_tiny_res50_wt_adam_001_cosine 2>&1 # this is current best
#python main.py --config configs/training/resnet50/tiny_resnet50_training_adam_0001_cosine.yml > log_tiny_res50_wt_adam_0001_cosine 2>&1 # this is current best
#python main.py --config configs/training/resnet50/tiny_resnet50_training_sgd_multi.yml #> log_tiny_res50_wt_sgd_multi 2>&1 # this is current best




#python main.py --config configs/training/resnet50/tiny_resnet50_training_adam_short.yml > log_tiny_res50_wt_adam_short 2>&1 # this is current best
#python main.py --config configs/training/resnet50/tiny_resnet50_training_adam.yml > log_tiny_res50_wt_adam 2>&1 # this is current best




# TinyImageNet, ResNet-101

## Weight training
#python main.py --config configs/training/resnet101/tiny_resnet101_training.yml > log_tiny_res101_wt_adam 2>&1 # this is current best
#python main.py --config configs/training/resnet101/tiny_resnet101_training_300.yml > log_tiny_res101_wt_adam_300 2>&1 # this is current best

#python main.py --config configs/training/resnet101/tiny_resnet101_training.yml > log_tiny_res101_wt 2>&1 # this is current best


## HC
#python main.py --config configs/hypercube/tinyImageNet/resnet101/resnet101_sparsity_5.yml  > log_tiny_res101_hc_sparsity_5 2>&1


## EP
#python main.py --config configs/ep/tinyImageNet/resnet101/resnet101_sparsity_5.yml > log_tiny_res101_ep_sparsity_5 2>&1
#python main.py --config configs/ep/tinyImageNet/resnet101/resnet101_sparsity_50.yml > log_tiny_res101_ep_sparsity_50 2>&1






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

#python main.py --config configs/hypercube/tinyImageNet/resnet18/resnet18_sparsity_50_adam.yml > log_tiny_hc_sparsity_50_adam 2>&1
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
