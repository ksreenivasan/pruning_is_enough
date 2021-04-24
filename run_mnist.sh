:<<BLOCK
python mnist_pruning_exps.py --mode pruning \
--algo ep \
--lr 0.1 \
--optimizer sgd \
--wd 0.0005 \
--epochs 50 \
--results-filename resuls_acc_mnist_ep_sgd.csv
BLOCK


:<<BLOCK
python mnist_pruning_exps.py --mode pruning \
--algo hc \
--lr 0.01 \
--optimizer adam \
--wd 0 \
--epochs 50 \
--results-filename resuls_acc_mnist_hc_noreg_adam.csv
BLOCK

#<<BLOCK
python mnist_pruning_exps.py --mode pruning \
--algo hc \
--lmbda 0.000001 \
--regularization var_red_1 \
--lr 0.01 \
--optimizer adam \
--wd 0 \
--epochs 50 \
--results-filename resuls_acc_mnist_hc_reg_adam.csv
#BLOCK


<<BLOCK
python mnist_pruning_exps.py --mode training \
--lr 0.001 \
--wd 0.0001 \
--optimizer adam \
--epochs 50 \
--results-filename resuls_acc_mnist_weight_train_adam.csv
BLOCK
