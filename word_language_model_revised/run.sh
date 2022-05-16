export CUDA_VISIBLE_DEVICES=3

# train from scratch
#python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --tied --epochs 40 > log_scratch_training 2>&1


# renda, prune-rate 0.1
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 \
            --tied --algo renda --rounds 20 --epochs 40 --prune_rate 0.1 > log_renda_40ep_20round_prune_rate_0_1 2>&1

# imp, prune-rate 0.1
# python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 \
#             --tied --algo imp --rounds 20 --epochs 40 --prune_rate 0.1 --not_finalized #> log_imp_40ep_20round_prune_rate_0_1 2>&1

