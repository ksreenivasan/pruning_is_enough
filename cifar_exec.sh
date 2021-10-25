#export cuda_visible_devices=2

#### ResNet-18
#python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml #> log_resnet18_ep 2>&1
python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml #> log_resnet18_hc 2>&1
#python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml #> log_resnet18_hc 2>&1







#### Conv4

# Weight training (WT)
#python main.py --config configs/training/conv4/conv4_training.yml

# EP
#python main.py --config configs/ep/conv4/conv4_sc_ep.yml 


