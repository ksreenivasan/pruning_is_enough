# ===== cold long IMP ===== #

subfd="long_cold_imp"
n_gpu=3

# python imp_main.py \
# --config configs/imp/resnet20.yml \
# --imp_rewind_iter 0 \
# --gpu $n_gpu \
# --subfolder $subfd

for i in 29 28 24 23 19 18 14 13 7 3 1 0 27 26 25 22 21 20 17 16 15 12 11 10 9 8 6 5 4 2
do
    python imp_sanity.py \
    --config configs/imp/resnet20.yml \
    --subfolder $subfd \
    --imp-resume-round $i \
    --imp-rewind-model results/$subfd/Liu_checkpoint_model_correct.pth \
    --gpu $n_gpu
done

