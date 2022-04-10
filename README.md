## Rare Gems: Finding Lottery Tickets at Initialization

### Overview
---
It has been widely observed that large neural networks can be pruned to a small fraction of their original size, with little loss in accuracy, by typically following a time-consuming "train, prune, re-train" approach. Frankle & Carbin (2018) conjecture that we can avoid this by training lottery tickets, i.e., special sparse subnetworks found at initialization, that can be trained to high accuracy. However, a subsequent line of work presents concrete evidence that current algorithms for finding trainable networks at initialization, fail simple baseline comparisons, e.g., against training random sparse subnetworks. Finding lottery tickets that train to better accuracy compared to simple baselines remains an open problem. In this work, we partially resolve this open problem by discovering rare gems: subnetworks at initialization that attain considerable accuracy, even before training. Refining these rare gems - "by means of fine-tuning" - beats current baselines and leads to accuracy competitive or better than magnitude pruning methods.

### Depdendencies (tentative)
---
Tested stable depdencises:
* python 3.6.5 (Anaconda)
* PyTorch 1.1.0
* torchvision 0.2.2
* CUDA 10.0.130
* cuDNN 7.5.1
* tensorboard
* tqdm
* ffcv (If you want to run ffcv imagenet)

### Data Preparation
---
1. For `tinyimagenet`, run `load_tiny_imagenet.sh`
2. For `imagenet`, you will need to download `imagenet` and specify the path in `data/imagenet.py` (Currently in branch. Will be merged soon)

### Running Experients:
---
The main script is `main.py`, to launch the jobs, we provide scripts `./cifar_exec.sh`, `imp_exec.sh`. And we provide a description of the main arguments. For more detailed descriptions, refer to `args_helper.py`.


| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `algo` | Specify the algorithm to run. `hc|ep|hc_iter|wt`. Note that GM is `hc_iter` in the code. |
| `lr` | Inital learning rate that will be used for the pruning process. |
| `fine_tune_lr` | Inital learning rate that will be used for the finetuning process. |
| `batch-size` | Batch size for the optimizers e.g. SGD or Adam. |
| `optimizer` | `sgd` or `adam`. |
| `dataset`      | Dataset to use. |
| `arch`      | Model to use. |
| `gamma` | the factor of learning rate decay, i.e. the effective learning rate is `lr*gamma^t`. |
| `iter_period` | Specifically for `hc_iter`, how often to run iterative thresholding. |
| `conv_type` | Will almost always be `SubnetConv` for pruning. |
| `target_sparsity` | Specify the target sparsity for the ticket. |
| `unflag_before_finetune` | Restore weights if the regularizer killed too many. |
| `init` | Weight initialization distribution. |
| `score_init` | Score initialization distribution. |
| `hc_quantized` | Enable for GM since it will round on forward pass. |
| `regularization` | `L2|L1` |
| `lmbda` | Regularization weight. |
| `gpu` | Specify which gpu to run on. |


#### Configs
Note that the workflow is managed by specifying the above arguments using `.yml` files specified in the `configs/` directory. Please refer them to create new configs like `configs/resnet20/resnet20_sparsity_0_59_unflagT.yml`.

#### Sample Config
```
# subfolder: target_sparsity_0_59_unflagT

# Hypercube optimization
algo: 'hc_iter'
iter_period: 5

# Architecture
arch: resnet20

# ===== Dataset ===== #
dataset: CIFAR10
name: resnet20_quantized_iter_hc

# ===== Learning Rate Policy ======== #
optimizer: sgd
lr: 0.1 #0.01
lr_policy: cosine_lr #constant_lr #multistep_lr
fine_tune_lr: 0.01
fine_tune_lr_policy: multistep_lr

# ===== Network training config ===== #
epochs: 150 
wd: 0.0 
momentum: 0.9
batch_size: 128

# ===== Sparsity =========== #
conv_type: SubnetConv
bn_type: NonAffineBatchNorm
freeze_weights: True
prune_type: BottomK
# enter target sparsity here
target_sparsity: 0.59
# decide if you want to "unflag"
unflag_before_finetune: True
init: signed_constant
score_init: unif #skew #half #bimodal #skew # bern
scale_fan: False #True

# ===== Rounding ===== #
round: naive 
noise: True
noise_ratio: 0 

# ===== Quantization ===== #
hc_quantized: True
quantize_threshold: 0.5

# ===== Regularization ===== #
regularization: L2
lmbda: 0.0001 # 1e-4

# ===== Hardware setup ===== #
workers: 4
gpu: 0

# ===== Checkpointing ===== #
checkpoint_at_prune: False

# ==== sanity check ==== #
skip_sanity_checks: False

```