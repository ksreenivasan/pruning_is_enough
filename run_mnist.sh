:<<BLOCK
python mnist_pruning_exps.py --mode pruning \
--algo hc \
--lmbda 0.00000 \
--regularization \
--lr 0.01 \
--optimizer adam \
--wd 0 \
--epochs 50
BLOCK

python mnist_pruning_exps.py --mode pruning \
--algo hc \
--lmbda 0.000001 \
--regularization \
--lr 0.01 \
--optimizer adam \
--wd 0 \
--epochs 50
