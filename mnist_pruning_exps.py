# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import colors as mcolors
from pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

import pdb
import time
import copy
plt.style.use('seaborn-whitegrid')

glob_args = None

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, bias_scores, k):
        if glob_args.algo == 'pt_hack':
            # Get the supermask by normalizing scores and "sampling" by probability
            if glob_args.normalize_scores:
                # min-max normalization so that scores are in [0, 1]
                min_score = scores.min().item()
                max_score = scores.max().item()
                scores = (scores - min_score)/(max_score - min_score)

                # repeat for bias
                min_score = bias_scores.min().item()
                max_score = bias_scores.max().item()
                bias_scores = (bias_scores - min_score)/(max_score - min_score)

            ## sample using scores as probability
            ## by default the probabilities are too small. artificially
            ## pushing them towards 1 helps!
            MULTIPLIER = 10
            scores = torch.clamp(MULTIPLIER*scores, 0, 1)
            bias_scores = torch.clamp(MULTIPLIER*bias_scores, 0, 1)
            out = torch.bernoulli(scores)
            bias_out = torch.bernoulli(bias_scores)

        elif glob_args.algo == 'ep':
            # Get the supermask by sorting the scores and using the top k%
            out = scores.clone()
            _, idx = scores.flatten().sort()
            j = int((1 - k) * scores.numel())
            # flat_out and out access the same memory.
            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1

            # repeat for bias
            # Get the supermask by sorting the scores and using the top k%
            bias_out = bias_scores.clone()
            _, idx = bias_scores.flatten().sort()
            j = int((1 - k) * bias_scores.numel())

            # flat_out and out access the same memory.
            bias_flat_out = bias_out.flatten()
            bias_flat_out[idx[:j]] = 0
            bias_flat_out[idx[j:]] = 1

        elif glob_args.algo == 'pt':
            scores = torch.clamp(MULTIPLIER*scores, 0, 1)
            bias_scores = torch.clamp(MULTIPLIER*bias_scores, 0, 1)
            out = torch.bernoulli(scores)
            bias_out = torch.bernoulli(bias_scores)

        else:
            print("INVALID PRUNING ALGO")
            print("EXITING")
            exit()

        return out, bias_out

    @staticmethod
    def backward(ctx, g_1, g_2):
        # send the gradient g straight-through on the backward pass.
        return g_1, g_2, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if glob_args.bias:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_scores = nn.Parameter(torch.Tensor(1))
        if glob_args.algo in ['hc', 'hc_iter']:
            if glob_args.scores_init == 'bern':
                self.scores.data = torch.bernoulli(0.5 * torch.ones_like(self.scores.data))
                self.bias_scores.data = torch.bernoulli(0.5 * torch.ones_like(self.bias_scores.data))
            elif glob_args.scores_init == 'unif':
                nn.init.uniform_(self.scores, a=0.0, b=1.0)
                nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        if glob_args.bias:
            self.bias.requires_grad = False

    def forward(self, x):
        if glob_args.algo in ['hc', 'hc_iter']:
            # don't need a mask here. the scores are directly multiplied with weights
            self.scores.data = torch.clamp(self.scores.data, 0.0, 1.0)
            self.bias_scores.data = torch.clamp(self.bias_scores.data, 0.0, 1.0)
            subnet = self.scores
            bias_subnet = self.bias_scores
        else:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), glob_args.sparsity)

        w = self.weight * subnet
        if glob_args.bias:
            b = self.bias * bias_subnet
        else:
            b = self.bias
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if glob_args.bias:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_scores = nn.Parameter(torch.Tensor(1))
        if glob_args.algo in ['hc', 'hc_iter']:
            if glob_args.scores_init == 'bern':
                self.scores.data = torch.bernoulli(0.5 * torch.ones_like(self.scores.data))
                self.bias_scores.data = torch.bernoulli(0.5 * torch.ones_like(self.bias_scores.data))
            elif glob_args.scores_init == 'unif':
                nn.init.uniform_(self.scores, a=0.0, b=1.0)
                nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        if glob_args.bias:
            self.bias.requires_grad = False

    def forward(self, x):
        if glob_args.algo in ['hc', 'hc_iter']:
            # don't need a mask here. the scores are directly multiplied with weights
            self.scores.data = torch.clamp(self.scores.data, 0.0, 1.0)
            self.bias_scores.data = torch.clamp(self.bias_scores.data, 0.0, 1.0)
            subnet = self.scores
            bias_subnet = self.bias_scores
        else:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), glob_args.sparsity)

        w = self.weight * subnet
        if glob_args.bias:
            b = self.bias * bias_subnet
        else:
            b = self.bias
        return F.linear(x, w, b)


# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SupermaskConv(1, 32, 3, 1, bias=glob_args.bias)
        self.conv2 = SupermaskConv(32, 64, 3, 1, bias=glob_args.bias)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = SupermaskLinear(9216, 128, bias=glob_args.bias)
        self.fc2 = SupermaskLinear(128, 10, bias=glob_args.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class NetNormal(nn.Module):
    # network for training
    def __init__(self):
        super(NetNormal, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=glob_args.bias)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=glob_args.bias)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128, bias=glob_args.bias)
        self.fc2 = nn.Linear(128, 10, bias=glob_args.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % glob_args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc = 100. * correct/len(test_loader.dataset)
    return test_acc


def get_layer_sparsity(layer):
    weight_mask, bias_mask = GetSubnet.apply(layer.scores.abs(), layer.bias_scores.abs(), glob_args.sparsity)
    weight_sparsity = 100.0 * weight_mask.sum().item() / weight_mask.flatten().numel()
    bias_sparsity = 100.0 * bias_mask.sum().item() / bias_mask.flatten().numel()
    # TODO: handle bias sparsity also
    return weight_sparsity #, bias_sparsity

def get_model_sparsity(model):
    # compute mean sparsity of each layer
    # TODO: find a nicer way to do this (skip dropout)
    s1 = get_layer_sparsity(model.conv1)
    s2 = get_layer_sparsity(model.conv2)
    s3 = get_layer_sparsity(model.fc1)
    s4 = get_layer_sparsity(model.fc2)

    avg_sparsity = (s1 + s2 + s3 + s4)/4
    return avg_sparsity

def get_model_sparsity_hc(model):
    # handle bias
    sparsity = []
    for name, params in model.named_parameters():
        if ".score" in name:
            num_middle = torch.gt(params, torch.ones_like(params)*0.00) * torch.lt(params, torch.ones_like(params)*1.00).int() # 0.25 / 0.75
            curr_sparsity = 100*torch.sum(num_middle).item()/num_middle.numel()
            sparsity.append(curr_sparsity)
            print(name, '{}/{} ({:.2f} %)'.format(torch.sum(num_middle).item(), num_middle.numel(), curr_sparsity))

    return sparsity

def get_score_sparsity_hc(model):
    sparsity = []
    numer = 0
    denom = 0
    for name, scores in model.named_parameters():
        if ".score" in name:
            num_ones = torch.sum(scores.detach().flatten())
            numer += num_ones.item()
            denom += scores.numel()
            curr_sparsity = 100 * num_ones.item()/scores.numel()
            sparsity.append(curr_sparsity)
            print(name, '{}/{} ({:.2f} %)'.format((int)(num_ones.item()), scores.numel(), curr_sparsity))
    print('overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))

    return 100*numer/denom



def compute_loss(model, device, train_loader, criterion):
    model.eval()

    '''
    for name, params in model.named_parameters():
        if ".score" in name:
            print(params[0][0][0][0])
            break
    '''

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target).detach().item()
        break
    return loss

def round_down(model, params, device, train_loader, criterion):

    scores = params.data
    scores2 = torch.ones_like(scores) * -1      # initialize a dummy tensor
    sc2 = scores2.flatten()

    # check indices I that has score value of neither 0 nor 1 
    sc = scores.flatten()
    flag_sc = torch.gt(sc, torch.zeros_like(sc)) * torch.lt(sc, torch.ones_like(sc)).int()

    # for i \in [n]/I, copy params values to dummy tensor   
    sc2[flag_sc == 0] = sc[flag_sc == 0]

    start_time = time.time()    
    # for i in I:
        # computes loss_1 & loss_0
        # depending on the difference, fill in a dummy tensor
    #temp = torch.clone(params.data.flatten())
    for idx in range(len(flag_sc)):

        if (idx+1) % 100 == 0:
            end_time = time.time()
            print(idx, end_time - start_time)

        if flag_sc[idx] == 1:
            
            #temp = torch.clone(params.data.flatten()[idx])
            #print(params.data[0][0][0][0])
            params.data.flatten()[idx] = 1
            #print(params.data[0][0][0][0])
            torch.manual_seed(idx)
            loss1 = compute_loss(model, device, train_loader, criterion)

            params.data.flatten()[idx] = 0
            #print(params.data[0][0][0][0])
            torch.manual_seed(idx)
            loss0 = compute_loss(model, device, train_loader, criterion)

            #print(loss1, loss0)

            if loss1 > loss0:   sc2[idx] = 0
            else:   sc2[idx] = 1

            params.data.flatten()[idx] = temp[idx]
            #print(params.data[0][0][0][0])
            #print(sum(scores2.flatten()))  
    
    #print(scores2.flatten())

    return scores2


def plot_histogram_scores(model, epoch=0):
    # TODO: make this generalizable
    plt.rcParams.update({'font.size': 5})
    fig, axs = plt.subplots(2, 2)
    scores = model.conv1.scores.flatten().cpu().detach().numpy()
    axs[0, 0].hist(scores, facecolor='#2ab0ff', edgecolor='#169acf',
                   density=False, linewidth=0.5, bins=20)
    axs[0, 0].set_title('Conv1 Scores Distribution')

    scores = model.conv2.scores.flatten().cpu().detach().numpy()
    axs[0, 1].hist(scores, facecolor='#2ab0ff', edgecolor='#169acf',
                   density=False, linewidth=0.5, bins=20)
    axs[0, 1].set_title('Conv2 Scores Distribution')

    scores = model.fc1.scores.flatten().cpu().detach().numpy()
    axs[1, 0].hist(scores, facecolor='#2ab0ff', edgecolor='#169acf',
                   density=False, linewidth=0.5, bins=20)
    axs[1, 0].set_title('FC1 Scores Distribution')

    scores = model.fc2.scores.flatten().cpu().detach().numpy()
    axs[1, 1].hist(scores, facecolor='#2ab0ff', edgecolor='#169acf',
                   density=False, linewidth=0.5, bins=20)
    axs[1, 1].set_title('FC2 Scores Distribution')

    filename = 'plots/weights_histogram_epoch_{}.pdf'.format(epoch)
    plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.05)


def round_and_evaluate(model, device, criterion, train_loader, test_loader):
    test(model, device, criterion, test_loader)
    acc_list = []
    score_sparsity_list = []
    for itr in range(1):
        #model.load_state_dict(torch.load('mnist_pruned_model_{}_{}.pt'.format(glob_args.algo, glob_args.epochs)))
        print('Testing rounding technique of {}'.format(glob_args.round))
        for name, params in model.named_parameters():
            if ".score" in name:
                if glob_args.round == 'naive':
                    #params.data = torch.gt(params, torch.ones_like(params)*0.5).int()
                    params.data = torch.gt(params.detach(), torch.ones_like(params.data)*0.5).int().float()
                elif glob_args.round == 'noisy':
                    params.data = torch.gt(params.detach(), torch.ones_like(params.data)*0.5).int().float()
                    pdb.set_trace()
                    params.data += torch.bernoulli(torch.ones_like(params.detach().data)*0.01)
                    params.data = torch.fmod(params.detach().data, 2).int()
                elif glob_args.round == 'prob':
                    #print(torch.bernoulli(params.detach()[0]))
                    #pdb.set_trace()
                    params.data = torch.bernoulli(params.detach().data)
                elif glob_args.round == 'pb':
                    params.data = round_down(model, params, device, train_loader, criterion)
                    print(name, ' ended')
                else:
                    print("INVALID ROUNDING")
                    print("EXITING")
                    exit()

        acc = test(model, device, criterion, test_loader)
        acc_list = np.append(acc_list, np.array([acc]))
        score_sparsity = get_score_sparsity_hc(model)
        score_sparsity_list = np.append(score_sparsity_list, np.array([score_sparsity]))

    print("Rounding results: ")
    print('Mean Acc: {}, Std Dev: {}'.format(np.mean(acc_list), np.std(acc_list)))
    print('Mean Sparsity: {}, Std Dev: {}'.format(np.mean(score_sparsity_list), np.std(score_sparsity_list)))
    #torch.save(model.state_dict(), "mnist_pruned_model_{}_{}_{}_{}.pt".format(glob_args.algo, glob_args.epochs, glob_args.scores_init, glob_args.round))
    #pdb.set_trace()

    return acc_list



def main():
    global glob_args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='../data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    parser.add_argument('--p-threshold', type=float, default=0.05,
                        help='probability threshold for pruning')
    parser.add_argument('--normalize-scores', action='store_true', default=False,
                        help='to normalize or not to normalize')
    parser.add_argument('--results-filename', type=str, default='results_acc_mnist.csv',
                        help='csv results filename')
    parser.add_argument('--lmbda', type=float, default=0.001,
                        help='regularizer coefficient lambda')
    # ep: edge-popup, pt_hack: KS hacky probability pruning, pt_reg: probability pruning with regularization
    # hc: hypercube pruning
    parser.add_argument('--algo', type=str, default='ep',
                         help='pruning algo to use |ep|pt_hack|pt_reg|hc|hc_iter|')
    parser.add_argument('--optimizer', type=str, default='sgd',
                         help='optimizer option to use |sgd|adam|')
    parser.add_argument('--evaluate-only', action='store_true', default=False,
                        help='just use rounding techniques to evaluate a saved model')
    parser.add_argument('--round', type=str, default='naive',
                         help='rounding technique to use |naive|prob|pb|') # naive: threshold(0.5), prob: probabilistic rounding, pb: pseudo-boolean paper's choice (RoundDown)
    parser.add_argument('--scores-init', type=str, default='unif', 
                        help='rounding technique to use |unif|bern|') # unif: unif(0,1), bern: bern(0.5)
    parser.add_argument('--num-test', type=int, default=1,
                        help='number of different models testing in prob rounding')
    parser.add_argument('--mode', type=str, default="pruning",
                        help='can be used for either pruning | training.')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='can be used for either pruning | training.')
    parser.add_argument('--pretrained', type=str, default=None,
                         help='path of pretrained model')
    epoch_list = []
    test_acc_list = []
    model_sparsity_list = []

    glob_args = parser.parse_args()
    use_cuda = not glob_args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(glob_args.seed)

    device = torch.device("cuda:2" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(glob_args.data, 'mnist'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=glob_args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(glob_args.data, 'mnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=glob_args.test_batch_size, shuffle=True, **kwargs)

    if glob_args.mode == "pruning":
        model = Net().to(device)
    elif glob_args.mode == "training":
        model = NetNormal().to(device)
    else:
        raise NotImplementedError("Non-supported mode ...")

    if glob_args.pretrained is not None:
        model.load_state_dict(torch.load(glob_args.pretrained))

    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    if glob_args.optimizer == 'sgd':
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=glob_args.lr,
            momentum=glob_args.momentum,
            weight_decay=glob_args.wd,
        )

    elif glob_args.optimizer == 'adam':
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                     lr=glob_args.lr,
                     weight_decay=glob_args.wd,
                     amsgrad=False)
    else:
        print("INVALID OPTIMIZER")
        print("EXITING")
        exit()

    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=glob_args.epochs)

    if not glob_args.evaluate_only:
        for epoch in range(1, glob_args.epochs + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            #if epoch % 50 == 0 and glob_args.algo in ['hc_iter']: # 50
            #if epoch % 1 == 0:
            #if epoch >= 20 and epoch % 5 == 0 and glob_args.algo in ['hc_iter']: # 50
                #pdb.set_trace()
            #    round_and_evaluate(model, device, criterion, train_loader, test_loader)

            test_acc = test(model, device, criterion, test_loader)
            scheduler.step()
            epoch_list.append(epoch)
            test_acc_list.append(test_acc)
            if glob_args.mode != "training":
                if glob_args.algo in ['hc', 'hc_iter']:
                    model_sparsity = get_model_sparsity_hc(model)
                else:
                    model_sparsity = get_model_sparsity(model)

                #if epoch%10 == 1:
                #    plot_histogram_scores(model, epoch)
            else:
                model_sparsity = (sum([p.numel() for p in model.parameters()]))

            model_sparsity_list.append(model_sparsity)
            print("Test Acc: {:.2f}%\n".format(test_acc))
            print("---------------------------------------------------------")
            results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc': test_acc_list, 'model_sparsity': model_sparsity_list})
            results_df.to_csv(glob_args.results_filename, index=False)

        if glob_args.mode != "training":
            # gotta plot the final histogram as well
            plot_histogram_scores(model, epoch)

        if glob_args.save_model:
            if glob_args.mode != 'training':
                if glob_args.algo in ['hc', 'hc_iter']:
                    model_filename = "mnist_pruned_model_{}_{}_{}.pt".format(glob_args.algo, glob_args.epochs, glob_args.scores_init)
                else:
                    model_filename = "mnist_pruned_model_{}_{}.pt".format(glob_args.algo, glob_args.epochs)
            else:
                model_filename = "mnist_trained_model_{}.pt".format(glob_args.epochs)
            torch.save(model.state_dict(), model_filename)

    if glob_args.algo in ['hc', 'hc_iter']:
        # irrespective of evaluate_only, add an evaluate_only step
        model.load_state_dict(torch.load('mnist_pruned_model_{}_{}_{}.pt'.format(glob_args.algo, glob_args.epochs, glob_args.scores_init)))
        orig_model = copy.deepcopy(model)
        print('I am here')
        for itr in range(glob_args.num_test):
            model = copy.deepcopy(orig_model)
            round_and_evaluate(model, device, criterion, train_loader, test_loader)
        #print("Test Acc: {:.2f}%\n".format(round_acc_list))

    print("Experiment donezo")

if __name__ == '__main__':
    main()
