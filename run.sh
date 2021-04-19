#python mnist_pruning_exps.py --algo hc --lr 0.001 --optimizer adam --save-model --epochs 50  # train


#python mnist_pruning_exps.py --algo hc --lr 0.001 --optimizer adam --train 0 --epochs 50 --round pb --batch-size 1 #> log_rounddown 2>&1
#python mnist_pruning_exps.py --algo hc --lr 0.001 --optimizer adam --train 0 --epochs 50 --round prob --num_test 100



# CIFAR-10, train (weight training)
:<<BLOCK
python main.py --config configs/training/conv4/conv4_training.yml \
                --name conv4-adam-training \
                --fixed-init \
                --seed 1224
BLOCK

# # CIFAR-10, test (weight training), mode connectivity
:<<BLOCK
python main.py --config configs/training/conv4/conv4_training.yml \
               --pretrained 'models/pretrained/conv4-weight-training-fixed-init-seed-1218.pth' \
               --pretrained2 'models/pretrained/conv4-weight-training-fixed-init-seed-1224.pth' \
               --evaluate \
               --fixed-init \
               --seed 42  #\
#               --interpolate linear
BLOCK



## CIFAR-10, train (EP)
:<<BLOCK
python main.py --config configs/ep/conv4/conv4_sc_ep.yml \
               --name conv4-sc-ep \
               --prune-rate 0.5 \
               --fixed-init \
               --seed 1424
BLOCK



# CIFAR-10, test (EP)
#:<<BLOCK
#python main.py --config configs/hypercube/conv4/conv4_kn_unsigned.yml \
#               --name conv4-kn-unsigned \
#               --prune-rate 0.5 \
#               --pretrained 'models/pretrained/conv4-kn-unsigned.pth' \
#               --evaluate 1
#BLOCK



# # CIFAR-10, train (HC)
:<<BLOCK
python main.py  --config configs/hypercube/conv4/conv4_sc_no_lr_decay_hypercube.yml \
                --name conv4-sc-bern-no-lr-decay-hypercube \
                --prune-rate 0.0 \
                --score-init bern \
                --hc-warmup 30 \
                --hc-period 10 \
                --round naive \
                --seed 2038 \
                --fixed-init \
                --noise \
                --noise-ratio 0.00 
BLOCK

# # CIFAR-10, test (HC), mode connectivity
#:<<BLOCK
python main.py  --config configs/hypercube/conv4/conv4_sc_hypercube.yml \
                --prune-rate 0.0 \
                --pretrained 'models/pretrained/conv4-sc-no-lr-decay-hypercube-149-fixed-init-seed-1532.state' \
                --pretrained2 'models/pretrained/conv4-sc-no-lr-decay-hypercube-149-fixed-init-seed-1957.state' \
                --evaluate \
                --fixed-init \
                --seed 42 \
                --round naive \
                --noise \
                --noise-ratio 0.0 
#BLOCK
#                --pretrained 'models/pretrained/conv4-sc-no-lr-decay-hypercube-149.state' \
#                --pretrained2 'models/pretrained/conv4-sc-no-lr-decay-hypercube-149-seed-1532.state' \



# # CIFAR-10, test (HC), epoch-30 model (81.77%)
:<<BLOCK
python main.py --config configs/hypercube/conv4/conv4_sc_hypercube.yml \
                --prune-rate 0.0 \
                --pretrained 'models/pretrained/conv4-sc-no-lr-decay-hypercube-30.pth' \
                --evaluate \
                --round naive \
                --noise \
                --noise-ratio 0.0 
BLOCK



# # CIFAR-10, test (HC), epoch-1 model
:<<BLOCK
python main.py --config configs/hypercube/conv4/conv4_sc_hypercube.yml \
                --prune-rate 0.0 \
                --pretrained 'models/pretrained/conv4-sc-no-lr-decay-hypercube-0.pth' \
                --evaluate \
                --round naive \
                --noise \
                --noise-ratio 0.0 
BLOCK

