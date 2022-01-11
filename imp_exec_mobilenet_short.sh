

# ===== warm short IMP ===== #

subfd="short_warm_imp_mobilenet"
n_gpu=2

python imp_main.py \
--config configs/imp/mobilenet.yml \
--iter_period 15 \
--epochs 300 \
--gpu $n_gpu \
--subfolder $subfd

# for sanity check
for i in 19 18 14 13 7 3 1 0 17 16 15 12 11 10 9 8 6 5 4 2
do
    python imp_sanity.py \
    --config configs/imp/mobilenet.yml \
    --epochs 300 \
    --subfolder $subfd \
    --imp-resume-round $i \
    --imp-rewind-model results/$subfd/Liu_checkpoint_model_correct.pth \
    --gpu $n_gpu
done

