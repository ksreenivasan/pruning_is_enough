conf_end=".yml"
log_end="_log"

conf_file="configs/training/lstm/wt"
log_root="lstm_wiki_test_wt"
subfolder_root="lstm_wiki_test_wt"

for trial in 1
do
    python main.py \
    --gpu 1 \
    --config "$conf_file$conf_end" \
    --trial-num $trial \
    --use-full-data \
    --subfolder "$subfolder_root$trial" #> "$log_root$trial$log_end" 2>&1 &

done
