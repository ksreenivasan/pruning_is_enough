from functools import partial
import os
import pdb
import pathlib
import shutil
import math
import numpy as np
import random

import torch
import torch.nn as nn


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
