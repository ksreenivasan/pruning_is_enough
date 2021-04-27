
#:<<BLOCK
seed=(1 2)
for s in ${seed[@]}
do
    python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_fixed_init.yml --seed $s
done
#BLOCK




