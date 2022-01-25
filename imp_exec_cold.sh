# ===== cold short IMP ===== #

subfd="short_cold_imp"
n_gpu=2

# python imp_main.py \
# --config configs/imp/resnet20.yml \
# --imp_rewind_iter 0 \
# --gpu $n_gpu \
# --subfolder $subfd

# for i in 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
for i in 27 22 21 20 19 18 17 0
do
    python imp_sanity.py \
    --config configs/imp/resnet20_tmp.yml \
    --subfolder $subfd \
    --imp-resume-round $i \
    --imp-rewind-model results/$subfd/Liu_checkpoint_model_correct.pth \
    --gpu $n_gpu
done



# IMP
:<<BLOCK
subfd = short_imp_sgd

python imp_main.py \
--arch resnet20 \
--dataset CIFAR10 \
--algo imp \
--optimizer sgd \
--lr 0.1 \
--lr-policy multistep_lr \
--lr-gamma 0.1 \
--epochs 150 \
--wd 1e-4 \
--momentum 0.9 \
--batch-size 128 \
--iter_period 5 \
--seed 42 \
--prune-rate 0.2 \
--imp_rewind_iter 1000 \
--conv-type SubnetConv \
--bn-type NonAffineBatchNorm \
--gpu 1 \
--init kaiming_normal \
--subfolder "$subfd/"
BLOCK

:<<BLOCK
#for i in {19}
#do
i=18
    python imp_sanity.py \
    --arch resnet20 \
    --dataset CIFAR10 \
    --algo imp \
    --optimizer sgd \
    --subfolder "$subfd" \
    --lr 0.1 \
    --lr-policy multistep_lr \
    --lr-gamma 0.1 \
    --epochs 150 \
    --wd 1e-4 \
    --momentum 0.9 \
    --batch-size 128 \
    --seed 42 \
    --conv-type SubnetConv \
    --bn-type NonAffineBatchNorm \
    --imp-resume-round $i \
    --imp-rewind-model results/$subfd/Liu_checkpoint_model_correct.pth \
    --gpu 2
#done
BLOCK
