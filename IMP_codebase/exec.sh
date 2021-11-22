# export cuda_visible_devices=3

####################### Basic schemes

# IMP
#:<<BLOCK
python cifar_lth.py \
--lr 0.1 \
--gamma 0.1 \
--epochs 3200 \
--optimizer sgd \
--momentum 0.9 \
--wd 1e-4 \
--rewind_iter 1000 \
--iter_period 160 \
--batch-size 128 \
--prune_perct 20 
#BLOCK

####################### HC (iterative)
:<<BLOCK
python mnist_lth.py \
--lr 0.001 \
--epochs 50 \
--optimizer adam \
--wd 0 \
--iter_period 5
BLOCK
