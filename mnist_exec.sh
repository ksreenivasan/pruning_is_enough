:<<BLOCK
# EP (10% sparsity)
python mnist_pruning_exps.py --algo ep --lr 0.01 --epochs 50 --optimizer sgd --sparsity 0.9 --results-filename vanilla_ep_sparsity_0.9.csv

# vanilla HC
python mnist_hc.py --algo hc --lr 0.01 --epochs 50 --optimizer adam --wd 0 --results-filename hc_vanilla_adam_lr_1e-2_50epoch.csv

# iterative HC (iter_period = 5)
python mnist_hc_iterative.py --algo hc_iter --lr 0.01 --epochs 50 --optimizer adam --wd 0 --iter_period 5 --results-filename hc_iter_period_5_adam_lr_1e-2_50epoch.csv

# iterative HC (iter_period = 1)
python mnist_hc_iterative.py --algo hc_iter --lr 0.01 --epochs 50 --optimizer adam --wd 0 --iter_period 1 --results-filename hc_iter_period_1_adam_lr_1e-2_50epoch.csv
BLOCK

# iterative HC + WT test
python mnist_hc_iterative.py --algo hc_iter --lr 0.01 --epochs 10 --optimizer adam --wd 0 --iter_period 1 --switch_to_wt 1 --switch_epoch 5 --results-filename hc_iter_switch_wt_test.csv

