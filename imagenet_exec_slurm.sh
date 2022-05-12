#!/bin/bash
#SBATCH --job-name=hongyiwa-test    # create a short name for your job
#SBATCH --output=hc_resnet50_imagenet_trial0.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=202:00:00          # total run time limit (HH:MM:SS)
 
source /apps/local/conda_init.sh
conda activate pf2.0

conf_file="configs/hypercube/resnet50/imagenet/resnet50_sparsity_5.yml"
log_root="resnet50_sp5_gpu_"
log_end="_log"

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --config "$conf_file"

#torchrun main.py --config "$conf_file"
#python main.py --rank 1 --config "$conf_file" &
#python main.py --rank 2 --config "$conf_file" &
#python main.py --rank 3 --config "$conf_file" &
