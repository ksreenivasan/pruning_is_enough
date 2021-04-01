from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
import os
import math
import random
import copy

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

# %matplotlib inline

sparsity=0.5
criterion = nn.BCELoss(reduce=False)

def seed_experiment(seed=0):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeded everything")

    
# define mean hinge loss
def hinge_loss(output, target):
    loss = 1 - torch.mul(output, target)
    # hingeloss = max(0, 1-yi*y_hat)
    loss = torch.max(torch.zeros(target.shape), loss)
    # loss[loss < 0] = 0
    return loss.mean()

class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the mask
        self.mask = torch.bernoulli(0.5 + torch.zeros_like(self.weight))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.mask.requires_grad = False

    def forward(self, x):
        w = self.weight * self.mask
        return F.linear(x, w.float(), self.bias)
        return x
    

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


# size of NN is k
class SimpleNN(nn.Module):

    def __init__(self, k=1):
        super(SimpleNN, self).__init__()
        # k leaky relu nodes
        self.fc1 = nn.Linear(2, k, bias=True)
        # vector v
        self.fc2 = nn.Linear(k, 1, bias=True)
        # initialize weights with zero as it is easy to track
        self.fc1.weight.data.fill_(0)
        self.fc2.weight.data.fill_(0)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# size of NN is k
class PruningSimpleNN(nn.Module):

    def __init__(self, k=1):
        super(PruningSimpleNN, self).__init__()
        # k leaky relu nodes
        self.fc1 = SupermaskLinear(2, k, bias=False)
        # vector v
        self.fc2 = SupermaskLinear(k, 1, bias=False)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x

def get_data():
    # generate synthetic data
    # balls around (5,0) and (-5,0)
    data = [
            (torch.tensor([15, 0], dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)),
            (torch.tensor([16, 0], dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)),
            (torch.tensor([14, 0], dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)),
            (torch.tensor([15, 1], dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)),
            (torch.tensor([15, -1], dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)),
             
            (torch.tensor([-15, 0], dtype=torch.float32), torch.tensor(-1., dtype=torch.float32)),
            (torch.tensor([-16, 0], dtype=torch.float32), torch.tensor(-1., dtype=torch.float32)),
            (torch.tensor([-14, 0], dtype=torch.float32), torch.tensor(-1., dtype=torch.float32)),
            (torch.tensor([-15, 1], dtype=torch.float32), torch.tensor(-1., dtype=torch.float32)),
            (torch.tensor([-15, -1], dtype=torch.float32), torch.tensor(-1., dtype=torch.float32)),]
 
    return data

def randomize_labels(data):
    randomized_data = []
    random_targets = np.random.choice([1, -1], size=len(data))
    for i, (data_i, target_i) in enumerate(data):
        randomized_data.append((data_i, torch.tensor(random_targets[i], dtype=torch.float32)))
    return randomized_data


def train_brute_force(model, data, batch_size=10, lr=0.1, momentum=0, MAX_ITER=100, debug_level=100):
    # SGD
    # optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    # loss function is hinge_loss
    trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=5)
    # 1 pass of data per mask
    min_loss = 100
    MAX_ITER = 100000
    for layer in list(model.children())[::-1]:
        print("Optimizing Layer: {}".format(layer))
        n = layer.mask.shape.numel()
        opt_mask = copy.deepcopy(layer.mask)
        for i in range(0, 2**n):
            if i > MAX_ITER:
                layer.mask = opt_mask
                break
            print("Binary Mask: {}".format(i))
            bin_str = np.binary_repr(i, width=n)
            mask_vec = torch.tensor(np.array(list(bin_str), dtype='float'))
            mask_vec = mask_vec.reshape(layer.mask.shape)
            layer.mask = mask_vec
            # check loss of each mask over all data
            loss = 0
            for j, data in enumerate(trainloader):
                input_j, target_j = data
                pred_j = torch.flatten(model(input_j))
                loss += hinge_loss(pred_j, target_j)

            print("Loss: {} | min_loss: {}".format(loss, min_loss))
            if loss <= min_loss:
                opt_mask = copy.deepcopy(layer.mask)
                min_loss = loss
                if loss <= 0.03:
                    # found optimal classifier
                    layer.mask = opt_mask
                    return
        # finally choose optimal mask
        layer.mask = opt_mask


def train_v1(model, data, batch_size=10, lr=0.1, momentum=0, MAX_ITER=100, debug_level=100):
    # SGD
    # optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    # loss function is hinge_loss
    trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=5)
    for epoch in range(MAX_ITER):
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            input_i, target_i = data
            # do a pass over every weight, in every layer
            for layer in model.children():
                for act_idx in range(len(layer.mask)):
                    for weight_idx in range(len(layer.mask[act_idx])):
                        # flick it on
                        # TODO: make this work for higher dimensions as well
                        layer.mask[act_idx][weight_idx] = torch.ones_like(layer.mask[act_idx][weight_idx])
                        pred_i = torch.flatten(model(input_i))
                        on_loss = hinge_loss(pred_i, target_i).mean()

                        # flick it off
                        layer.mask[act_idx][weight_idx] = torch.zeros_like(layer.mask[act_idx][weight_idx])
                        pred_i = torch.flatten(model(input_i))
                        off_loss = hinge_loss(pred_i, target_i).mean()
                        # print("on_loss={} | off_loss={}".format(on_loss, off_loss))

                        if on_loss >= off_loss:
                            layer.mask[act_idx] = torch.ones_like(layer.mask[act_idx])
                            epoch_loss = on_loss
                        else:
                            layer.mask[act_idx] = torch.zeros_like(layer.mask[act_idx])
                            epoch_loss = off_loss
            
        if epoch%(debug_level) == 0:
            print("Epoch={}, Loss={}".format(epoch, epoch_loss))


def train(model, data, batch_size=10, lr=0.1, momentum=0, MAX_ITER=100, debug_level=100):
    # SGD
    # optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    # loss function is hinge_loss
    trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=5)
    for epoch in range(MAX_ITER):
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            input_i, target_i = data
            # do a pass over every weight, in every layer
            for layer in model.children():
                for act_idx in range(len(layer.mask)):
                    # flick it on
                    # TODO: make this work for higher dimensions as well
                    layer.mask[act_idx] = torch.ones_like(layer.mask[act_idx])
                    pred_i = torch.flatten(model(input_i))
                    on_loss = hinge_loss(pred_i, target_i).mean()

                    # flick it off
                    layer.mask[act_idx] = torch.zeros_like(layer.mask[act_idx])
                    pred_i = torch.flatten(model(input_i))
                    off_loss = hinge_loss(pred_i, target_i).mean()
                    # print("on_loss={} | off_loss={}".format(on_loss, off_loss))

                    if on_loss >= off_loss:
                        layer.mask[act_idx] = torch.ones_like(layer.mask[act_idx])
                        epoch_loss = on_loss
                    else:
                        layer.mask[act_idx] = torch.zeros_like(layer.mask[act_idx])
                        epoch_loss = off_loss
            
        if epoch%(debug_level) == 0:
            print("Epoch={}, Loss={}".format(epoch, epoch_loss))


# data is [(input, label)]
def get_gaussian_data(x1=(25,0), x2=(-25, 0), std=2, lin_sep=True, n=100):
    # generate 2n data points drawn from gaussian centered at x1 and x2
    data = []
    # positive data centered around x1
    for i in range(n):
        x = torch.empty(2).normal_(mean=0,std=std) + torch.tensor(x1)
        if lin_sep:
            if x[0].item() < 0:
                continue
        data.append((x, torch.tensor(1, dtype=torch.float32)))

    # negative labels centered around x2
    for i in range(n):
        x = torch.empty(2).normal_(mean=0,std=std) + torch.tensor(x2)
        if lin_sep:
            if x[0].item() > 0:
                continue
        data.append((x, torch.tensor(-1, dtype=torch.float32)))

    return data

def plot_decision_boundary(model, data, filename="myplot.png"):
    x_vals = np.linspace(-50, 50, 30)
    y_vals = np.linspace(-50, 50, 30)
    Z = np.zeros([30, 30])
    X, Y = np.meshgrid(x_vals, y_vals)
    for i in range(30):
        for j in range(30):
            # print i, j
            x = torch.tensor([X[i][j], Y[i][j]], dtype=torch.float32)
            Z[i][j] = model.forward(x)

    predictions = {'x_0': [], 'x_1': [], 'y': []}
    for i in range(30):
        for j in range(30):
            predictions['x_0'].append(X[i][j])
            predictions['x_1'].append(Y[i][j])
            predictions['y'].append(Z[i][j])
    prediction_df = pd.DataFrame(predictions)

    training_data = {'x_0': [], 'x_1': [], 'y': []}
    for (data_inp, label) in data:
        training_data['x_0'].append(data_inp[0].item())
        training_data['x_1'].append(data_inp[1].item())
        training_data['y'].append(1.0*label.item())
    training_data = pd.DataFrame(training_data)

    ax = prediction_df[prediction_df['y'] >= 0].plot.scatter(x='x_0', y='x_1', color='red', label='y=+1')
    prediction_df[prediction_df['y'] < 0].plot.scatter(x='x_0', y='x_1', color='blue', label='y=-1', ax=ax)
    # overlay training data
    training_data[training_data['y'] >= 0].plot.scatter(x='x_0', y='x_1', color='orange', label='training_data', ax=ax, marker='*', s=500)
    training_data[training_data['y'] < 0].plot.scatter(x='x_0', y='x_1', color='purple', label='training_data', ax=ax, marker='*', s=500)
    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.05)

def get_test_accuracy(model, data):
    loss = 0
    for i, (data_i, target_i) in enumerate(data):
        pred_i = torch.flatten(model(data_i))
        loss += hinge_loss(pred_i, target_i)
    print("Mean Hinge Loss={}".format(loss.item()/len(data)))


# seed things
# seed_experiment(42)

# synthetic data
print("Synthetic data training")
model = PruningSimpleNN(k=10)
# print(model.fc1.weight)
data_org = get_data()
train_brute_force(model, data_org, batch_size=10, lr=0.01, MAX_ITER=100, debug_level=1)
### plot
plot_decision_boundary(model, data_org, "synth_data.png")
get_test_accuracy(model, data_org)

# gaussian data
print("Gaussian data training")
model = PruningSimpleNN(k=10)
data_org = get_gaussian_data(std=10, n=5)
train_brute_force(model, data_org, lr=0.1, batch_size=10, MAX_ITER=100, debug_level=10)
### plot
plot_decision_boundary(model, data_org, "clean_label_1.png")
get_test_accuracy(model, data_org)

# random label training
print("Random Label training")
# model = PruningSimpleNN(k=100)
data_rand = randomize_labels(data_org)
train_brute_force(model, data_rand, lr=0.001, batch_size=10, MAX_ITER=200, debug_level=10)
plot_decision_boundary(model, data_rand, "rand_label.png")
get_test_accuracy(model, data_rand)

# clean training to finish it off
print("Clean label tuning")
train(model, data_org, lr=0.1, batch_size=10, MAX_ITER=100, debug_level=10)
plot_decision_boundary(model, data_org, "clean_label_2.png")
get_test_accuracy(model, data_org)

