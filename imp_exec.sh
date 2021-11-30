
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
--batch_size 128 \
--iter_period 8 \
--seed 42 \
--prune_rate 0.2 \
--imp_rewind_iter 1000 \
--conv_type SubnetConv \
--bn_type NonAffineBatchNorm \
--gpu 1 \
--subfolder "short_imp_sgd/"
#BLOCK

#:<<BLOCK
for i in {0..19}
do 
    python imp_sanity.py \
    --arch resnet20 \
	--dataset CIFAR10 \
    --optimizer sgd \
    --subfolder "short_imp_sgd" \
    --lr 0.1 \
    --lr-policy multistep_lr \
    --lr-gamma 0.1 \
    --epochs 150 \
    --wd 1e-4 \
    --momentum 0.9 \
    --batch_size 128 \
    --seed 42 \
    --conv_type SubnetConv \
	--bn_type NonAffineBatchNorm \
    --imp-resume-round $i \
    --imp-rewind-model short_imp_sgd/Liu_checkpoint_model_correct.pth \
    --gpu 2
done
#BLOCK