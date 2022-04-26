# ResNet50_tf (changed the output size)

:<<BLOCK
# Running trials in parallel
conf_file="configs/final_hc/caltech_2.yml"
log_root="hc_caltech_2_rebuttal_"
log_end="_log"
subfolder_root="hc_caltech_2_rebuttal_"

for trial in 1 2 3
do
#    python main.py \
#    --config "$conf_file" --subfolder "$subfolder_root$trial" \
#    --trial-num "$trial" > "$log_root$trial$log_end" 2>&1 #&

    python main.py \
    --config "$conf_file" --subfolder "invert_$subfolder_root$trial" \
    --trial-num "$trial" --invert-sanity-check --skip-sanity-checks > "invert_$log_root$trial$log_end" 2>&1 #&
done


conf_file="configs/final_hc/caltech_5.yml"
log_root="hc_caltech_5_rebuttal_"
log_end="_log"
subfolder_root="hc_caltech_5_rebuttal_"


for trial in 1 2 3
do
    python main.py \
    --config "$conf_file" --subfolder "invert_$subfolder_root$trial" \
    --trial-num "$trial" --invert-sanity-check --skip-sanity-checks > "invert_$log_root$trial$log_end" 2>&1 #&
done
BLOCK




#:<<BLOCK
# IMP
subfd="long_warm_imp_transfer"
n_gpu=0

python imp_main.py \
--config configs/imp/transfer_1FC.yml \
--epochs 5000 \
#--gpu $n_gpu \
#--subfolder $subfd
#BLOCK

# Weight training
#python main.py --config configs/training/resnet50/caltech_resnet50_training_1FC.yml #> log_caltech_wt_50epoch_1FC 2>&1
#python main.py --config configs/training/resnet50/caltech_resnet50_training_2FC.yml #> log_caltech_wt_50epoch_2FC 2>&1



# 1FC
:<<BLOCK
python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_50_1FC.yml > log_caltech_hc_sparsity_50_1FC 2>&1
python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_5_1FC.yml > log_caltech_hc_sparsity_5_1FC 2>&1
python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_2_1FC.yml > log_caltech_hc_sparsity_2_1FC 2>&1
python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_50_1FC.yml > log_caltech_ep_sparsity_50_1FC 2>&1
python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_5_1FC.yml > log_caltech_ep_sparsity_5_1FC 2>&1
python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_2_1FC.yml > log_caltech_ep_sparsity_2_1FC 2>&1
BLOCK


# 2FC
:<<BLOCK
python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_50_2FC.yml > log_caltech_hc_sparsity_50_2FC 2>&1
python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_5_2FC.yml > log_caltech_hc_sparsity_5_2FC 2>&1
python main.py --config configs/hypercube/resnet50/caltech101/caltech101_resnet50_hc_sparsity_2_2FC.yml > log_caltech_hc_sparsity_2_2FC 2>&1
python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_50_2FC.yml > log_caltech_ep_sparsity_50_2FC 2>&1
python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_5_2FC.yml > log_caltech_ep_sparsity_5_2FC 2>&1
python main.py --config configs/ep/resnet50/caltech101/caltech101_resnet50_ep_sparsity_2_2FC.yml > log_caltech_ep_sparsity_2_2FC 2>&1
BLOCK







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

