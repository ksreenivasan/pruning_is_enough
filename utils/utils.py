from functools import partial
import os
import pdb
import pathlib
import shutil
import math
import numpy as np

import torch
import torch.nn as nn
import matplotlib as plt
from matplotlib import colors as mcolors
from pylab import *
import random
plt.style.use('seaborn-whitegrid')


# set seed for experiment
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # making sure GPU runs are deterministic even if they are slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeded everything: {}".format(seed))


def plot_histogram_scores(model, algo='undefined', epoch=0):
    # TODO: make this generalizable
    plt.rcParams.update({'font.size': 5})
    n_row, n_col = 3, 3
    fig, axs = plt.subplots(n_row, n_col)

    idx = 0
    for name, params in model.named_parameters():
        if ".score" in name:
            scores = params.data.flatten().cpu().numpy()
            r, c = divmod(idx, n_row)
            axs[r, c].hist(scores, facecolor='#2ab0ff', edgecolor='#169acf',
                   density=False, linewidth=0.5, bins=20)
            axs[r, c].set_title('{}'.format(name))                   
            idx += 1

    filename = 'plots/weights_histogram_{}_epoch_{}.pdf'.format(algo, epoch)
    plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.05)
