#python mnist_pruning_exps.py --algo hc --lr 0.001 --optimizer adam --save-model --epochs 50  # train


#python mnist_pruning_exps.py --algo hc --lr 0.001 --optimizer adam --train 0 --epochs 50 --round pb --batch-size 1 #> log_rounddown 2>&1
#python mnist_pruning_exps.py --algo hc --lr 0.001 --optimizer adam --train 0 --epochs 50 --round prob --num_test 100


# CIFAR-10, test (EP)
#:<<BLOCK
#python main.py --config configs/hypercube/conv4/conv4_kn_unsigned.yml \
#               --multigpu 1,2 \
#               --name conv4-kn-unsigned \
#               --prune-rate 0.5 \
#               --pretrained 'models/pretrained/conv4-kn-unsigned.pth' \
#               --evaluate 1
#BLOCK

## CIFAR-10, train (EP)
## Note: EP seems to do worse in the beginning with bias
#python main.py --config configs/hypercube/conv4/conv4_kn_ep.yml \
#              --multigpu 1,2 \
#              --name conv4-kn-ep \
#              --prune-rate 0.5 \
#              --bias \


# # CIFAR-10, train (HC)
:<<BLOCK
python main.py --config configs/hypercube/conv4/conv4_sc_no_lr_decay_hypercube.yml \
                --multigpu 1,2 \
                --name conv4-sc-bern-no-lr-decay-hypercube \
                --prune-rate 0.0 \
                --score-init bern \
                --hc-warmup 30 \
                --hc-period 10 \
                --round naive \
                --seed 1532 \
                --noise \
                --noise-ratio 0.00 
BLOCK

# # CIFAR-10, test (HC), mode connectivity
#:<<BLOCK
python main.py --config configs/hypercube/conv4/conv4_sc_hypercube.yml \
                --multigpu 0,3 \
                --prune-rate 0.0 \
                --pretrained 'models/pretrained/conv4-sc-no-lr-decay-hypercube-149.state' \
                --pretrained2 'models/pretrained/conv4-sc-no-lr-decay-hypercube-149-seed-1532.state' \
                --evaluate \
                --round naive \
                --noise \
                --noise-ratio 0.0 
#BLOCK

# # CIFAR-10, test (HC), epoch-30 model (81.77%)
:<<BLOCK
python main.py --config configs/hypercube/conv4/conv4_sc_hypercube.yml \
                --multigpu 1,2 \
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
                --multigpu 0,3 \
                --prune-rate 0.0 \
                --pretrained 'models/pretrained/conv4-sc-no-lr-decay-hypercube-0.pth' \
                --evaluate \
                --round naive \
                --noise \
                --noise-ratio 0.0 
BLOCK


