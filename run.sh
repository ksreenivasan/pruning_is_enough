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


# # CIFAR-10, test (HC)
 python main.py --config configs/hypercube/conv4/conv4_kn_hypercube.yml \
                --multigpu 1,2 \
                --name conv4-kn-unsigned-hypercube-adam-0.01--scale-fan_False \
                --prune-rate 0.0 \
#                --pretrained 'models/pretrained/conv4-kn-unsigned-hypercube-adam-0.01--scale-fan_False.pth' \
#                --evaluate 1
#                --bias \


## CIFAR-10, train (EP)
## Note: EP seems to do worse in the beginning with bias
python main.py --config configs/hypercube/conv4/conv4_kn_ep.yml \
              --multigpu 1,2 \
              --name conv4-kn-ep \
              --prune-rate 0.5 \
#              --bias \
