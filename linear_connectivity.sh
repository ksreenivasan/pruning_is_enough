
seed=(1 2)
filename='models/pretrained/conv4-sc-hypercube-seed'

#:<<BLOCK
for s in ${seed[@]}
do
    python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_mode_connect.yml \
        --seed $s --mode-connect-filename $filename-$s.state
done
#BLOCK

###### Need to move the model to the pretrained directories

#:<<BLOCK
python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_mode_connect.yml \
                --pretrained $filename-1.state \
                --pretrained2 $filename-2.state \
                --evaluate 
#BLOCK

