
# IMP
#:<<BLOCK
python imp_main.py \
--arch resnet20 \
--dataset CIFAR10 \
--optimizer sgd \
--lr 0.1 \
--lr-policy multistep_lr \
--lr-gamma 0.1 \
--epochs 160 \
--wd 1e-4 \
--momentum 0.9 \
--batch-size 128 \
--iter_period 8 \
--seed 42 \
--prune-rate 0.2 \
--imp_rewind_iter 1000 \
--conv-type SubnetConv \
--bn-type AffineBatchNorm \
--gpu 1 \
--init kaiming_normal \
--subfolder "short_imp_sgd_normal/"
#BLOCK

#:<<BLOCK
for i in {0..19}
do 
    python imp_sanity.py \
    --arch resnet20 \
    --dataset CIFAR10 \
    --optimizer sgd \
    --subfolder "short_imp_sgd_normal" \
    --lr 0.1 \
    --lr-policy multistep_lr \
    --lr-gamma 0.1 \
    --epochs 150 \
    --wd 1e-4 \
    --momentum 0.9 \
    --batch-size 128 \
    --seed 42 \
    --conv-type SubnetConv \
    --bn-type AffineBatchNorm \
    --imp-resume-round $i \
    --imp-rewind-model results/short_imp_sgd_normal/Liu_checkpoint_model_correct.pth \
    --gpu 2
done
#BLOCK
