
# ===== warm short IMP ===== #

subfd="long_imp_no_rewind"
n_gpu=3

python imp_main.py \
--config configs/imp/resnet20_no_epochs.yml \
--epochs 4800 \
--iter_period 160 \
--imp-no-rewind \
--gpu $n_gpu \
--subfolder $subfd

# for i in 29 28 24 23 19 18 14 13 7 3 1 0 27 26 25 22 21 20 17 16 15 12 11 10 9 8 6 5 4 2
# do
#     python imp_sanity.py \
#     --config configs/imp/resnet20_no_epochs.yml \
#     --epochs 150 \
#     --imp-no-rewind \
#     --subfolder $subfd \
#     --imp-resume-round $i \
#     --gpu $n_gpu
# done

