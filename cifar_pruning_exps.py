# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

import pdb
import torchvision
from models.frankle import Conv4
from utils.net_utils import set_model_prune_rate

from args import args

#from utils.builder import get_builder

#args = None

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, bias_scores, k):
        if args.algo == 'pt_hack':
            # Get the supermask by normalizing scores and "sampling" by probability
            if args.normalize_scores:
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

        elif args.algo == 'ep':
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

        elif args.algo == 'pt':
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
        self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        if args.algo in ('hc'):
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.bias_scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        if args.algo in ('hc'):
            # don't need a mask here. the scores are directly multiplied with weights
            self.scores.data = torch.clamp(self.scores.data, 0.0, 1.0)
            self.bias_scores.data = torch.clamp(self.bias_scores.data, 0.0, 1.0)
            subnet = self.scores
            bias_subnet = self.bias_scores
        else:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), sparsity)

        w = self.weight * subnet
        b = self.bias * bias_subnet
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        if args.algo in ('hc'):
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.bias_scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        if args.algo in ('hc'):
            # don't need a mask here. the scores are directly multiplied with weights
            self.scores.data = torch.clamp(self.scores.data, 0.0, 1.0)
            self.bias_scores.data = torch.clamp(self.bias_scores.data, 0.0, 1.0)
            subnet = self.scores
            bias_subnet = self.bias_scores
        else:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), sparsity)

        w = self.weight * subnet
        b = self.bias * bias_subnet
        return F.linear(x, w, b)


# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

'''
class Conv4(nn.Module):
    denf __init__(self):
        super(Conv4, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(32 * 32 * 8, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 8192, 1, 1)
        out = self.linear(out)
        return out.squeeze()
'''

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
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
    mask = GetSubnet.apply(layer.scores.abs(), layer.bias_scores.abs(), 0)
    sparsity = 100.0 * mask.sum().item() / mask.flatten().numel()
    return sparsity

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
    sparsity = []
    for name, params in model.named_parameters():
        if ".score" in name:
            num_middle = torch.gt(params, torch.ones_like(params)*0.01) * torch.lt(params, torch.ones_like(params)*0.99).int() # 0.25 / 0.75
            curr_sparsity = 100*torch.sum(num_middle).item()/num_middle.numel()
            sparsity.append(curr_sparsity)
            print(name, '{}/{} ({:.2f} %)'.format(torch.sum(num_middle).item(), num_middle.numel(), curr_sparsity))

    return sparsity


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

'''
def round_down(model, params):
    scores = torch.clone(params).detach()
    flat_sc = scores.flatten()
    b_sc = torch.gt(scores, torch.zeros_like(scores)) * torch.lt(scores, torch.ones_like(scores)).int()
    flat_b_sc = b_sc.flatten()
    while sum(flat_b_sc) > 0:
        for idx in range(len(flat_b_sc)):
            if flat_b_sc[idx] == 1:
                cp_flat_sc_1 = torch.clone(flat_sc)
                cp_flat_sc_1[idx] = 1
                cp_flat_sc_0 = torch.clone(flat_sc)
                cp_flat_sc_0[idx] = 0

                pdb.set_trace()
                delta = loss_1 - loss_0
                exit()
        
        exit()
    return b_sc
'''


def plot_histogram_scores(model, epoch=0):
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
    training_data[training_data['y'] >= 0].plot.scatter(x='x_0', y='x_1', color='orange', label='training_data', ax=ax, marker='*', s=250)
    training_data[training_data['y'] < 0].plot.scatter(x='x_0', y='x_1', color='purple', label='training_data', ax=ax, marker='*', s=250)
    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.05)



def main():
    #global args
    # Training settings  

    '''
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
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='../data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    parser.add_argument('--p-threshold', type=float, default=0.05,
                        help='probability threshold for pruning')
    parser.add_argument('--normalize-scores', action='store_true', default=True,
                        help='to normalize or not to normalize')
    parser.add_argument('--results-filename', type=str, default='results_acc_mnist.csv',
                        help='csv results filename')
    parser.add_argument('--lmbda', type=float, default=0.001,
                        help='regularizer coefficient lambda')
    # ep: edge-popup, pt_hack: KS hacky probability pruning, pt_reg: probability pruning with regularization
    # hc: hypercube pruning
    parser.add_argument('--algo', type=str, default='ep',
                         help='pruning algo to use |ep|pt_hack|pt_reg|hc|')
    parser.add_argument('--optimizer', type=str, default='sgd',
                         help='optimizer option to use |sgd|adam|')
    parser.add_argument('--train', type=int, default=1,
                        help='train a new model (default: 1)')
    parser.add_argument('--round', type=str, default='naive',
                         help='rounding technique to use |naive|prob|pb|') # naive: threshold(0.5), prob: probabilistic rounding, pb: pseudo-boolean paper's choice (RoundDown)
    parser.add_argument('--num_test', type=int, default=1,
                        help='number of different models testing in prob rounding')
    '''

    #args = parser.parse_args()

    epoch_list = []
    test_acc_list = []
    model_sparsity_list = []

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:2" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(args.data_dir, 'cifar10'), train=True, download=True, transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(args.data_dir, 'cifar10'), train=False, download=True, transform=transform_test)


    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset
        ,
        batch_size=args.batch_size, shuffle=False, **kwargs)



    model = Conv4().to(device)
    set_model_prune_rate(model, args.prune_rate)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )

    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                     lr=args.lr,
                     weight_decay=0,
                     amsgrad=False)
    else:
        print("INVALID OPTIMIZER")
        print("EXITING")
        exit()

    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)


    if not args.train: 
        model.load_state_dict(torch.load('saved_mnist_cnn_{}_{}.pt'.format(args.algo, args.num_epochs)))
        #get_model_sparsity_hc(model)
        test(model, device, criterion, test_loader)     
    
        cp_model = Conv4().to(device)
        acc_list = []
        for itr in range(args.num_test):
            
            cp_model.load_state_dict(torch.load('saved_mnist_cnn_{}_{}.pt'.format(args.algo, args.num_epochs)))
            print('Testing rounding technique of {}'.format(args.round))

            for name, params in cp_model.named_parameters():
                if ".score" in name:
                    if args.round == 'naive':
                        params.data = torch.gt(params, torch.ones_like(params)*0.5).int()   
                    elif args.round == 'prob':
                        params.data = torch.bernoulli(params)   
                    elif args.round == 'pb':
                        scores = params.data
                        # initialize a dummy tensor
                        scores2 = torch.ones_like(scores) * -1
                        sc2 = scores2.flatten()
                
                        # check indices I that has score value of neither 0 nor 1 
                        sc = scores.flatten()
                        flag_sc = torch.gt(sc, torch.zeros_like(sc)) * torch.lt(sc, torch.ones_like(sc)).int()
                        # for i \in [n]/I, copy params values to dummy tensor   
                        sc2[flag_sc==0] = sc[flag_sc==0]
                
                        # for i in I:
                            # computes loss_1 & loss_0
                            # depending on the difference, fill in a dummy tensor
                        for idx in range(len(flag_sc)):
                            if flag_sc[idx] == 1:
                                
                                temp = torch.clone(params.data.flatten()[idx])
                                #print(params.data[0][0][0][0])
                                params.data.flatten()[idx] = 1
                                #print(params.data[0][0][0][0])
                                torch.manual_seed(idx)
                                loss1 = compute_loss(cp_model, device, train_loader, criterion)

                                params.data.flatten()[idx] = 0
                                #print(params.data[0][0][0][0])
                                torch.manual_seed(idx)
                                loss0 = compute_loss(cp_model, device, train_loader, criterion)

                                print(loss1, loss0)

                                if loss1 > loss0:   sc2[idx] = 0
                                else:   sc2[idx] = 1

                                params.data.flatten()[idx] = temp
                                #print(params.data[0][0][0][0])
                                print(sum(scores2.flatten()))

                                #pdb.set_trace()
                        
                        
                        #pdb.set_trace()
                        print(scores2.flatten())

                        params.data = scores2   

                        #params.data = round_down(cp_model, params) 
                        #exit()
                    else:
                        print("INVALID ROUNDING")
                        print("EXITING")
                        exit()

            acc = test(cp_model, device, criterion, test_loader)        
            acc_list = np.append(acc_list, np.array([acc]))

        print('mean: {}, std: {}'.format(np.mean(acc_list), np.std(acc_list)))

        print('Test ended')
        exit()


    for epoch in range(1, args.num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test_acc = test(model, device, criterion, test_loader)
        scheduler.step()
        epoch_list.append(epoch)
        test_acc_list.append(test_acc)
        
        if not args.algo == 'ep':
            if args.algo == 'hc':
                model_sparsity = get_model_sparsity_hc(model)
            else:
                model_sparsity = get_model_sparsity(model)
            model_sparsity_list.append(model_sparsity)
            if epoch%5 == 0:
                plot_histogram_scores(model)

        print("Test Acc: {:.2f}%\n".format(test_acc))
        # print("Model Sparsity: {:.2f}%\n\n".format(model_sparsity))
        print("---------------------------------------------------------")

    results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc': test_acc_list, 'model_sparsity': model_sparsity_list})
    results_df.to_csv(args.results_filename, index=False)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn_{}_{}.pt".format(args.algo, args.num_epochs))

    print("Experiment donezo")

if __name__ == '__main__':
    main()
