#### ResNet-18

# Running trials in parallel
# NOTE: make sure to delete/comment subfolder from the config file or else it may not work
:<<BLOCK
conf_file="configs/hypercube/wideresnet28/wideresnet28_weight_training.yml"
log_root="wideresnet28_"
log_end="_log"
subfolder_root="wideresnet28_results_trial_"

for trial in 2 3
do
    python main.py \
    --config "$conf_file" \
    --trial-num $trial \
    --subfolder "$subfolder_root$trial" > "$log_root$trial$log_end" 2>&1 &

    python main.py \
    --config "$conf_file" \
    --trial-num $trial \
    --invert-sanity-check \
    --subfolder "invert_$subfolder_root$trial" > "invert_$log_root$trial$log_end" 2>&1 &
done

BLOCK

conf_file="configs/hypercube/resnet50/ffcv_imagenet/resnet50_sparsity_20.yml"
log_root="resnet50_ffcv_sp20"
log_end="_log"
python main.py \
    --config "$conf_file" > "$log_root$log_end" 2>&1 &
