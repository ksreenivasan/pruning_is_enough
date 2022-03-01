
# target sparsity 0.5

# python main.py \
# --config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_50.yml \
# --subfolder tiny_mobile_target_sparsity_50 \
# --gpu 0 &

# python main.py \
# --config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_5.yml \
# --subfolder tiny_mobile_target_sparsity_5 \
# --gpu 1 &

python main.py \
--config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_2_5.yml \
--subfolder tiny_mobile_target_sparsity_2_5 \
--gpu 0 &

python main.py \
--config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_1_4.yml \
--subfolder tiny_mobile_target_sparsity_1_4 \
--gpu 1 &

# python main.py \
# --config configs/hypercube/tinyImageNet/mobilenetV2/sparsity_1_4_tmp.yml \
# --subfolder cifar_mobile_target_sparsity_1_4 \
# --gpu 1