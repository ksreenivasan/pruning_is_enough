#### ResNet-18
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml # 93.17% at 150 epoch
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg.yml 
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_v2.yml 
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml 
# python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_evaluate.yml 


### ResNet-20
:<<BLOCK
# python main.py \
#    --config configs/hypercube/resnet20/resnet20_random_subnet.yaml > kartik_log 2>&1
BLOCK

#:<<BLOCK
#python main.py --config configs/hypercube/resnet20/resnet20_sc_hypercube_reg_GD.yml
#BLOCK

# attempt at getting 1.4% sparsity with 80% acc
 python main.py \
     --config configs/hypercube/resnet20/resnet20_quantized_hypercube_reg_bottom_K.yml > cifar_log 2>&1

# add score rewinding
# python main.py \
#    --config configs/hypercube/resnet20/resnet20_quantized_hypercube_reg_bottom_K_rewind.yml


# python main.py --config configs/hypercube/resnet20/resnet20_wt.yml

:<<BLOCK
python main.py \
    --config configs/ep/resnet18/resnet18_sc_ep.yml > cifar_log 2>&1
BLOCK

#### Conv4

# Weight training (WT)
#python main.py --config configs/training/conv4/conv4_training.yml

# EP
#python main.py --config configs/ep/conv4/conv4_sc_ep.yml 


