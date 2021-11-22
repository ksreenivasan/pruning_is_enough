#!/bin/bash

####################### Basic schemes

# IMP
#:<<BLOCK
for i in {0..19}
#for i in 0
do 
    python cifar_sanity.py \
    --dest-dir "short_imp" \
    --lr 0.1 \
    --gamma 0.1 \
    --epochs 5 \
    --optimizer sgd \
    --momentum 0.9 \
    --wd 1e-4 \
    --batch-size 128 \
    --resume-round $i \
    --gpu 2 \
    --rewind-model short_imp/Liu_checkpoint_model_correct.pth
done
