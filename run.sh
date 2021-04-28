# Conventional (Weight training, EP)
# python main.py --config configs/training/conv4/conv4_training.yml

# run EP/HC over multiple overparameterization setup (w/ SGD)
#:<<BLOCK
width_arr=(2) #1.5 2)
for th in ${width_arr[@]}
do
    #python main.py --config configs/ep/conv4/conv4_sc_ep_sgd.yml --width $th
    python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_adam.yml --width $th #> log_$th 2>&1
done
#BLOCK

# HC + regularization experiments 
# python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_adam.yml > log_entropy 2>&1

# for testing probabilistic pruning (for some layer) and naive rounding (for other layers)
# python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_reg_test_hybrid_pruning.yml


# run for various weight/score init and width
:<<BLOCK
weight_init_arr=('signed_constant') #('kaiming_normal') #('signed_constant' 'kaiming_normal')
score_init_arr=('unif') #('bern') #('unif') #('bern' 'unif')
#width_arr=(1.5 2)
for w in ${weight_init_arr[@]}
do
    for s in ${score_init_arr[@]}
    do
        for t in ${width_arr[@]}
        do
            python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_reg_for_high_before_rounding.yml --init $w --score-init $s --width $t #> log_noreg 2>&1
        done
    done
done
BLOCK


# Grid search on hyperparameter (for HC)
:<<BLOCK
lr_arr=(0.005 0.0075 0.01 0.02)
lmbda_arr=(0.000001 0.000002)

for lr in ${lr_arr[@]}
do
    for lm in ${lmbda_arr[@]}
    do
        python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_reg_for_high_before_rounding.yml --lr $lr --lmbda $lm
    done
done
BLOCK


# Compare naive rounding and probabilistic rounding
# python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_reg_compare_naive_prob.yml


#####################################################
# RESNET18 CIFAR10 EXPERIMENTS ######################
#####################################################

# python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml
python main.py --config configs/ep/conv4/conv4_sc_ep.yml
