#export cuda_visible_devices=2

 ### ResNet-20
:<<BLOCK
 python main.py \
     --config configs/hypercube/resnet20/resnet20_random_subnet.yaml > kartik_log 2>&1
python main.py \
     --config configs/hypercube/resnet20/resnet20_quantized_hypercube_reg.yml #> cifar_run_log 2>&1
BLOCK

python main.py \
     --config configs/hypercube/resnet20/resnet20_quantized_hypercube_reg_GD.yml #> cifar_run_log 2>&1


