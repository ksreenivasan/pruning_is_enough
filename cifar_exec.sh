#### ResNet-18
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml # 93.17% at 150 epoch
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg.yml 
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_v2.yml 
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml 
# python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml
# python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_iter_reg_evaluate.yml 


### ResNet-20

# use cosine lr 300 epochs, 150 train, 150 cont train
# python main.py \
#    --config configs/hypercube/resnet20/config10.yml \
#    --HC-cont-use-previous-optimizer \
#    --epochs 150 \
#    --HC-cont-epochs 150 \
#    --results-filename 1

# use cosine lr 150 epochs, then set cont lr to be original lr / 2, again 150 epochs for cosine lr
python main.py \
   --config configs/hypercube/resnet20/config10.yml \
   --HC-cont-lr 0.05 \
   --epochs 150 \
   --HC-cont-epochs 150 \
   --HC-cont-lr-policy cosine_lr \
   --results-filename 2a

# same as file2, different start lr
# python main.py \
#    --config configs/hypercube/resnet20/config10.yml \
#    --HC-cont-lr 0.01 \
#    --epochs 150 \
#    --HC-cont-epochs 150 \
#    --HC-cont-lr-policy cosine_lr \
#    --results-filename 2b

# same as file2, different start lr
# python main.py \
#    --config configs/hypercube/resnet20/config10.yml \
#    --HC-cont-lr 0.005 \
#    --epochs 150 \
#    --HC-cont-epochs 150 \
#    --HC-cont-lr-policy cosine_lr \
#    --results-filename 2c

# use consine lr 150 epochs with eta_min set to somthing
# python main.py \
#    --config configs/hypercube/resnet20/config10.yml \
#    --cosine-lr-min 0.05 \
#    --epochs 150 \
#    --HC-cont-epochs 150 \
#    --HC-cont-lr 0.05 \
#    --HC-cont-lr-policy cosine_lr \
#    --results-filename 3a

# same as file3, different start lr
# python main.py \
#    --config configs/hypercube/resnet20/config10.yml \
#    --cosine-lr-min 0.01 \
#    --epochs 150 \
#    --HC-cont-epochs 150 \
#    --HC-cont-lr 0.01 \
#    --HC-cont-lr-policy cosine_lr \
#    --results-filename 3b

# same as file3, different start lr
# python main.py \
#    --config configs/hypercube/resnet20/config10.yml \
#    --cosine-lr-min 0.005 \
#    --HC-cont-lr 0.005 \
#    --epochs 150 \
#    --HC-cont-epochs 150 \
#    --HC-cont-lr-policy cosine_lr \
#    --results-filename 3c

# use cosine lr 150 epochs, then sgd
# python main.py \
#    --config configs/hypercube/resnet20/config10.yml \
#    --HC-cont-lr 0.1 \
#    --epochs 150 \
#    --HC-cont-epochs 150 \
#    --HC-cont-lr-policy multistep_lr \
#    --results-filename 4

# use cosine lr 150 epochs, then adam
# python main.py \
#    --config configs/hypercube/resnet20/config10.yml \
#    --HC-cont-optimizer adam \
#    --epochs 150 \
#    --HC-cont-epochs 150 \
#    --HC-cont-lr 0.001 \
#    --results-filename 5


:<<BLOCK
python main.py --config configs/hypercube/resnet20/resnet20_sc_hypercube_reg_GD.yml
BLOCK

# attempt at getting 1.4% sparsity with 80% acc
:<<BLOCK
python main.py \
--config configs/hypercube/resnet20/resnet20_quantized_hypercube_reg_bottom_K.yml > cifar_log 2>&1
BLOCK

# EP
# python main.py \
# --config configs/ep/resnet20/resnet20_global_ep_iter.yml > cifar_log 2>&1

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


