export cuda_visible_devices=2

####################### Basic schemes

#EP
:<<BLOCK
python mnist_pruning_exps.py --algo ep \
--lr 0.01 \
--epochs 50 \
--optimizer sgd \
--sparsity 0.9 \
--results-filename vanilla_ep_sparsity_0.9.csv
BLOCK

# HC
:<<BLOCK
# HC
python mnist_pruning_exps.py --algo hc \
--lr 0.01 \
--epochs 50 \
--optimizer adam \
--wd 0 \
--results-filename vanilla_hc_adam.csv
BLOCK

####################### HC (iterative)
#:<<BLOCK
python mnist_hc_iterative.py --algo hc_iter \
--lr 0.01 \
--epochs 100 \
--optimizer adam \
--wd 0 \
--iter_period 5 \
--results-filename hc_iter_test.csv
#--results-filename hc_iter_period_5_adam_lr_1e-2_100epoch.csv
#BLOCK
:<<BLOCK
python mnist_hc_iterative.py --algo hc_iter \
--lr 0.01 \
--epochs 50 \
--optimizer adam \
--wd 0 \
--iter_period 1 \
--rewind 1 \
--results-filename hc_rewind_iter_period_1.csv
#--arch FC \
#--n_hidden_layer 3 \
BLOCK
#:<<BLOCK
python mnist_hc_iterative.py --algo hc \
--lr 0.01 \
--epochs 100 \
--optimizer adam \
--wd 0 \
--results-filename hc_vanilla_lr_1e-2_100epoch.csv
#BLOCK








####################### HC (activation pruning)

:<<BLOCK
python mnist_activation_hc.py --algo hc_act \
--lr 0.01 \
--epochs 50 \
--optimizer adam \
--wd 0 \
--arch FC \
--n_hidden_layer 3 \
BLOCK


### FC (2 layer)
:<<BLOCK
# Activation pruning (HC)
python mnist_activation_hc.py --algo hc_act \
--lr 0.1 \
--epochs 50 \
--optimizer sgd \
--arch FC \
--n_layer 2 \ 
#--results-filename pruning_hc_act_adam_FC.csv
BLOCK


### Ramanujan network
:<<BLOCK
# Activation pruning (HC)
python mnist_activation_hc.py --algo hc_act \
--lr 0.01 \
--epochs 50 \
--optimizer adam \
--wd 0
#--results-filename pruning_hc_act_adam.csv
BLOCK

:<<BLOCK
# Activation pruning (HC)
python mnist_activation_hc.py --algo hc_act \
--lr 0.01 \
--epochs 50 \
--optimizer adam \
--wd 0.0001 \
--bias \
--results-filename pruning_hc_act_adam.csv
BLOCK
