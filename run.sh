



#python mnist_pruning_exps.py --algo hc_iter --lr 0.01 --optimizer adam --wd 0 --scores-init unif --epochs 50 --round noisy # train & test for rounded version (iterative HC)
#python mnist_pruning_exps.py --algo hc_iter --lr 0.01 --optimizer adam --wd 0 --scores-init unif --epochs 50 --round naive --save-model # train & test for rounded version (iterative HC)
#python mnist_pruning_exps.py --algo hc_iter --pretrained saved_models_and_csv/mnist_pruned_model_hc_50_unif.pt --lr 0.01 --optimizer adam --wd 0 --scores-init unif --epochs 10 --round naive --save-model # train & test for rounded version (iterative HC)

#python mnist_pruning_exps.py --algo hc --lr 0.01 --optimizer adam --wd 0 --scores-init unif --save-model --epochs 50 --round naive # train & test for rounded version
#python mnist_pruning_exps.py --algo hc --lr 0.01 --optimizer adam --wd 0 --scores-init unif --evaluate-only --epochs 50 --round naive # only test
python mnist_pruning_exps.py --algo hc --lr 0.01 --optimizer adam --wd 0 --scores-init unif --epochs 50 --round prob --num-test 100





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
<< 'MULTILINE-COMMENT'
 python main.py --config configs/hypercube/conv4/conv4_kn_hypercube.yml \
                --multigpu 1,2 \
                --name conv4-kn-unsigned-hypercube-adam-0.01--scale-fan_False \
                --prune-rate 0.0 \
#                --pretrained 'models/pretrained/conv4-kn-unsigned-hypercube-adam-0.01--scale-fan_False.pth' \
#                --evaluate 1
#                --bias \
MULTILINE-COMMENT


## CIFAR-10, train (EP)
## Note: EP seems to do worse in the beginning with bias
#python main.py --config configs/hypercube/conv4/conv4_kn_unsigned.yml \
#              --multigpu 1,2 \
#              --name conv4-kn-unsigned \
#              --prune-rate 0.5 \
##              --bias \
