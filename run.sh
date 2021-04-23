#python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_prob.yml 
#python main.py --config configs/ep/conv4/conv4_sc_ep.yml > log_ep 2>&1
#python main.py --config configs/hypercube/conv4/conv4_sc_hypercube.yml > log_hc_noreg_naive 2>&1 
#python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_reg.yml #> log_hc_reg_naive 2>&1
python main.py --config configs/hypercube/conv4/conv4_sc_hypercube_reg_test_hybrid_pruning.yml 


