#export cuda_visible_devices=2

#### ResNet-18
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml # 93.17% at 150 epoch
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_v2.yml 
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml 
#python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_evaluate.yml 


### ResNet-20
# python main.py \
#     --config configs/hypercube/resnet20/resnet20_random_subnet.yaml > kartik_log 2>&1
# python main.py --config configs/hypercube/resnet20/resnet20_random_subnet.yaml
#python main.py --config configs/hypercube/resnet20/resnet20_wt.yml


#### Conv4

# Weight training (WT)
#python main.py --config configs/training/conv4/conv4_training.yml

# EP
#python main.py --config configs/ep/conv4/conv4_sc_ep.yml 


