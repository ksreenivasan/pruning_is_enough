#:<<BLOCK
python mnist_pruning_exps.py --mode pruning \
--algo hc \
--lr 0.01 \
--optimizer adam \
--wd 0 \
--epochs 50
#BLOCK

:<<BLOCK
python mnist_pruning_exps.py --mode pruning \
--algo hc \
--lmbda 0.000001 \
--regularization \
--lr 0.01 \
--optimizer adam \
--wd 0 \
--epochs 50
BLOCK
