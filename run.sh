#####################################################
# CONV4 CIFAR10 EXPERIMENTS #########################
#####################################################

# Conventional Weight training
:<<BLOCK
python main.py --config configs/training/conv4/conv4_training.yml > log_conv4_weight_training 2>&1
BLOCK

# Ramanujan's EP
:<<BLOCK
python main.py --config configs/ep/conv4/conv4_sc_ep.yml > log_conv4_sc_ep 2>&1
BLOCK


# run EP/HC over multiple overparameterization setup (w/ SGD)
:<<BLOCK
width_arr=(2) #1.5 2)
for th in ${width_arr[@]}
do
    # python main.py --config configs/ep/conv4/conv4_sc_ep_sgd.yml --width $th
    python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_adam.yml --width $th #> log_$th 2>&1
done
BLOCK

# HC + regularization experiments 
#:<<BLOCK
python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_reg_multistep_decay.yml > log_conv4_hc_reg_multistep 2>&1
#BLOCK

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
:<<BLOCK
python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_reg_compare_naive_prob.yml
BLOCK

#####################################################
# RESNET18 CIFAR10 EXPERIMENTS ######################
#####################################################

:<<BLOCK
python main.py --config configs/ep/resnet18/resnet18_sc_ep.yml > log_resnet18_ep 2>&1
python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_noreg.yml #> log_resnet18_hc 2>&1
BLOCK

:<<BLOCK
lmbda=(0.001) #0.01 0.0001 0.000001)
for lm in ${lmbda[@]}
do
    python main.py --config configs/hypercube/resnet18/resnet18_sc_hypercube_reg.yml --lmbda $lm # > log_resnet18_hc_lmbda_$lm 2>&1
done
BLOCK
