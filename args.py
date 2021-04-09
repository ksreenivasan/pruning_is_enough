import argparse
import sys
import yaml

from configs import parser as _parser

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pruning random networks")

    # Config/Hyperparameters
    parser.add_argument(
        "--data",
        default="data/datasets/",
        help="path to dataset base directory"
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Where to save the runs. If None use ./runs"   
        )
    parser.add_argument(
        "--name",
        default=None,
        type=str,
        help="Name of experiment"
    )
    parser.add_argument(
        "--config",
        default='configs/hypercube/conv4/conv4_kn_ep.yml',
        help="Config file to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of epochs to use in training (default: 10)"
    )
    parser.add_argument(
        "--submask-size",
        type=int,
        default=1,
        metavar="S",
        help="Size of random 0/1 submask to create among a random set of weights in the network (default: 1)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        metavar="M",
        help="Metric used to determine whether a weight or activation should be pruned (default: \"loss\")"
    )
    parser.add_argument(
        "--pruning-strategy",
        type=str,
        default=None,
        metavar="PS",
        help="Strategy for pruning. Can be \"weights\", \"activations\", \"activations_and_weights\", \"simulated_annealing\", or None. If None, train a model via vanilla SGD (default: None)"
    )
    parser.add_argument(
        "--how-to-prune",
        type=str,
        default="random",
        help="How the pruning strategy will be executed. Can be \"random\", \"random_with_replacement\", \"layer\", or \"layer_reversed.\" This argument has changes how the weights or activations in the network are selected for pruning. \"random\" will choose the weights/activations without replacement, \"random_with_replacement\" will choose the weights/activations with replacement, and \"layer\" will choose every weight/activation in the network in layer by layer, and \"layer_reversed\" is the same as \"layer,\" but will start with the last layer and move to the first (default: \"random\")"
    )
    parser.add_argument(
        "--start-from-nothing",
        action="store_true",
        default=False,
        help="Prunes all weights in the network, but leaves one weight per layer to connect the input layer to the output layer. Only applies to masked layers (default: False)"
    )
    parser.add_argument(
        "--flips",
        type=list,
        default=None,
        nargs="+",
        metavar="R",
        help="List of epoch indices (starting from and including 0) where flipping occurs. At each milestone, 10% of the mask parameters are randomly chosen and they are flipped. Restarting is accomplished at the beginning of an epoch. Applies only when pruning weights (default: None)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning rate (default: 0.001)"
    )

    parser.add_argument(
        "--lr-policy",
        type=str,
        default="cosine_lr",
        help="Learning rate scheduler"
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum (default: 0.9)"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0005,
        metavar="WD",
        help="Weight decay (default: 0.0005)"
    )
    parser.add_argument(
        "--nesterov",
        type=bool,
        default=False,
        metavar="N",
        help="Nesterov acceleration (default: False)"
    )
    parser.add_argument(
        "--milestones",
        type=list,
        default=[50, 100, 150, 200],
        nargs="+",
        metavar="M",
        help="List of milestones where learning rate is multiplied by args.gamma (default: [50, 100, 150, 200])"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        metavar="G",
        help="Multiplicative factor to reduce the learning rate at every milestone (default: 0.1)"
    )
    parser.add_argument(
        "--td",
        type=float,
        default=0.99,
        metavar="TD",
        help="Temperature decay constant for simulated annealing (default: 0.99)"
    )
    parser.add_argument(
        "--temp",
        type=int,
        default=100000,
        metavar="T",
        help="Temperature used in simulated annealing (default: 100000)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100000,
        metavar="MI",
        help="Maximum number of iterations to run simulated annealing for before terminating. It's recommended to set this to a value that at is at least as big as the number of parameters in the network (default: 100000)"
    )

    parser.add_argument(
        "--algo",
        type=str,
        default='ep',
        help="pruning algo to use |ep|pt_hack|pt_reg|hc|"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default='sgd',
        help="optimizer option to use |sgd|adam|"
    )

    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        default=False,
        help='just use rounding techniques to evaluate a saved model'
    )

    parser.add_argument(
        "--round",
        type=str,
        default='naive',
        help='rounding technique to use |naive|prob|pb|'
        # naive: threshold(0.5), prob: probabilistic rounding, pb: pseudo-boolean paper's choice (RoundDown)
    )

    parser.add_argument(
        '--noise',
        action='store_true',
        default=False,
        help='flag that decides if we add noise to the rounded p_i'
    )

    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.0,
        help="portion of score flipping"
    )

    parser.add_argument(
        "--score-init",
        type=str,
        default="unif",
        help="initial score for hypercube |unif|bern|"
    )

    parser.add_argument(
        "--num_test",
        type=int,
        default=1,
        help='number of different models testing in prob rounding'
    )

    parser.add_argument(
        "--save-model",
        action='store_true',
        default=False,
        help='For Saving the current Model'
    )


    # Architecture and training
    parser.add_argument(
        "--arch",
        type=str,
        default="TwoLayerFC",
        # KS: gotta find a better way to do this. causing circular import issues
        # help="Model architecture: " + " | ".join(models.__dict__["__all__"]) + " | (default: TwoLayerFC)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=500,
        metavar="H",
        help="Number of nodes in the FC layers of the network"
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Boolean flag to indicate whether to use bias"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        default=True,
        help="Boolean flag to indicate whether weights should be frozen. Used when sparsifying (default: False)"
    )

    parser.add_argument(
        "--conv-type",
        type=str,
        default=None,
        help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--mode",
        default="fan_in",
        help="Weight initialization mode"
    )
    parser.add_argument(
        "--nonlinearity",
        default="relu",
        help="Nonlinearity used by initialization"
    )
    parser.add_argument(
        "--bn-type",
        default=None,
        help="BatchNorm type"
    )
    parser.add_argument(
        "--init",
        default="kaiming_normal",
        help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay",
        action="store_true",
        default=False,
        help="No batchnorm decay"
    )
    parser.add_argument(
        "--scale-fan",
        action="store_true",
        default=False,
        help="scale fan"
    )
    parser.add_argument(
        "--first-layer-dense",
        action="store_true",
        help="First layer dense or sparse"
    )
    parser.add_argument(
        "--last-layer-dense",
        action="store_true",
        help="Last layer dense or sparse"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--first-layer-type",
        type=str,
        default=None,
        help="Conv type of first layer"
    )
    parser.add_argument(
        "--trainer", type=str, default="default", help="cs, ss, or standard training"
    )
    parser.add_argument(
        "--score-init-constant",
        type=float,
        default=None,
        help="Sample Baseline Subnet Init",
    )

    parser.add_argument(
        "--prune-rate",
        default=0.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="Dataset to train the model with. Can be CIFAR10 or MNIST (default: MNIST)"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cross-entropy-loss",
        help="which loss to use for pruning: cross-entropy-loss or zero-one-loss"
    )
    
#    # Save/Load
#    parser.add_argument(
#        "--save-dir",
#        type=str,
#        default=False,
#        help="Directory to save the results of the experiment (default: \"./results\")"
#    )
    parser.add_argument(
        "--save-plot-data",
        action="store_true",
        default=False,
        help="Boolean flag to indicate whether to save training data for plotting. The following data will be saved at every epoch: top-1 training set accuracy, top-1 validation set accuracy, epoch number (default: False)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory of dataset to load in"
    )
    parser.add_argument(
        "--load-ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to load from"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2,
        metavar="N",
        help="The number of batches to wait before logging training status. When pruning, this is the number of epochs before logging status (default: 100)"
    )
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=-1,
        help="The number of epochs to train before saving the next checkpoint"
    )

    # Device settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed (default: None)"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA training"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="G",
        help="Override the default choice for a CUDA-enabled GPU by specifying the GPU\"s integer index (i.e. \"0\" for \"cuda:0\")"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="W",
        help="Number of workers"
    )
    parser.add_argument(    
        "--shift",
        type=float,
        default=0.0,
        help="shift portion"
    )   
    parser.add_argument(    
        "--num-trial",
        type=int,
        default=1,
        help="number of trials for testing sharpness"
    )
    parser.add_argument(    
        "--multigpu",   
        default=None,   
        type=lambda x: [int(a) for a in x.split(",")],  
        help="Which GPUs to use for multigpu training", 
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    parser.add_argument(
        "--save_every",
        default=-1,
        type=int,
        help="Save every ___ epochs"
    )
    parser.add_argument(
        "--random-subnet",
        action="store_true",
        help="Whether or not to use a random subnet when fine tuning for lottery experiments",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
        )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--warmup_length",
        default=0,
        type=int,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        '--results-filename',
        type=str,
        default=None,
        help='csv results filename'
    )
    parser.add_argument(
        '--weight-training',
        action='store_true',
        default=False,
        help='flag that decides if we are doing pruning or weight training'
    )

    args = parser.parse_args()
    get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    args = parse_arguments()


run_args()

