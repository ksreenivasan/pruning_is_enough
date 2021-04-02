#python mnist_pruning_exps.py --algo hc --lr 0.001 --optimizer adam --save-model --epochs 50  # train


#python mnist_pruning_exps.py --algo hc --lr 0.001 --optimizer adam --train 0 --epochs 50 --round pb --batch-size 1 #> log_rounddown 2>&1
#python mnist_pruning_exps.py --algo hc --lr 0.001 --optimizer adam --train 0 --epochs 50 --round prob --num_test 100

python cifar_pruning_exps.py \
    --config configs/hypercube/conv4/conv4_kn_unsigned.yml \
    --lr 0.001 --optimizer adam --num-epochs 100 --round naive --prune-rate 0.5
    #--algo hc --lr 0.001 --optimizer adam --train 0 --num-epochs 50 --round naive --prune_rate 0.5
