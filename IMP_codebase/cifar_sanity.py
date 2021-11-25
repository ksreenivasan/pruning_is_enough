import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import random
import argparse
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.init as init
import pandas as pd
import numpy as np
import pickle
import torch
import copy
import sys
import os

from cifar_model_resnet import resnet20
from mask import Mask


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


def data_loader(parser_args):
    # get datapoints
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )
    train_dataset = torchvision.datasets.CIFAR10(
                        root=parser_args.data,
                        train=True,
                        download=True,
                        transform=transforms.Compose(
                            [
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]
                        ),
                    )
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=parser_args.batch_size, shuffle=True, **kwargs
                    )

    test_dataset = torchvision.datasets.CIFAR10(
                        root=parser_args.data,
                        train=False,
                        download=True,
                        transform=transforms.Compose([transforms.ToTensor(), normalize]),
                    )
    test_loader = torch.utils.data.DataLoader(
                        test_dataset, batch_size=parser_args.batch_size, shuffle=False, **kwargs
                    )

    return train_loader, test_loader


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            init.uniform_(m.weight, 0, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)


def redraw(model, mask, device, shuffle=False, reinit=False, chg_mask=False, chg_weight=False):
    cp_model = copy.deepcopy(model)
    cp_mask = copy.deepcopy(mask)
    for name, param in cp_model.named_parameters():
        if name in cp_model.prunable_layer_names:
            weight = param.data.detach()
            if shuffle:
                if chg_mask:
                    tmp_mask = cp_mask[()][name]
                    idx = torch.randperm(tmp_mask.nelement())
                    tmp_mask = tmp_mask.view(-1)[idx].view(tmp_mask.size())
                    mask[()][name] = tmp_mask
                if chg_weight:
                    idx = torch.randperm(weight.nelement())
                    weight = weight.view(-1)[idx].view(weight.size())
            elif reinit:
                init.kaiming_normal_(weight)
            elif not (shuffle and reinit):
                pass
            else:
                raise NotImplementedError
            param.data = weight * mask[()][name].to(device).float()
    return cp_model, mask


def sanity_check(parser_args, train_loader, test_loader, device, shuffle=False, reinit=False, chg_mask=False, chg_weight=False):
    """
    :param parser_args
    :param train_loader, test_loader
    :param device
    """

    # ======================================
    # =           Initialization           =
    # ======================================

    print("=================Use device {}===================".format(device))
    use_amp = True

    # load model
    model = resnet20(ignore_name=parser_args.ignore_name).to(device)
    # PATH_model = "model/round_{}_model.pth".format(parser_args.resume_round)
    PATH_model = parser_args.rewind_model
    checkpoint = torch.load(PATH_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    # load mask
    if parser_args.resume_round > 0:
        PATH_mask = "{}/round_{}_mask.npy".format(parser_args.dest_dir, parser_args.resume_round)
        mask = np.load(PATH_mask, allow_pickle=True)
    else:
        PATH_mask = "{}/round_1_mask.npy".format(parser_args.dest_dir)
        mask = np.load(PATH_mask, allow_pickle=True)
        for name in mask[()].keys():
            mask[()][name] = torch.ones_like(mask[()][name])

    # make change to mask or model
    model, mask = redraw(model, mask, device, shuffle, reinit, chg_mask, chg_weight)

    # Optimizer and criterion
    criterion = nn.CrossEntropyLoss().to(device)
    def get_optimizer_and_scheduler(parser_args):
        if parser_args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=parser_args.lr,
                momentum=parser_args.momentum,
                weight_decay=parser_args.wd,
            )
            # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if parser_args.gamma > 0.:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=parser_args.gamma)  # NOTE: Hard code
            else:
                scheduler = None

        elif parser_args.optimizer == 'adam':
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                         lr=parser_args.lr,
                                         weight_decay=parser_args.wd,
                                         amsgrad=False,
                                         )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler = None
        else:
            print("INVALID OPTIMIZER")
            print("EXITING")
            exit()
        return optimizer, scheduler

    optimizer, scheduler = get_optimizer_and_scheduler(parser_args)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


    # ======================================
    # =        Pre-define Function         =
    # ======================================
    
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = 100. * correct/len(test_loader.dataset)
        return test_acc

    
    def print_nonzeros(model, mask):
        nonzero = 0
        total = 0
        for name, p in model.named_parameters():
            if name in model.prunable_layer_names:
                tensor = p.data.detach().cpu().numpy()
                nz_count = np.count_nonzero(tensor)
                total_params = np.prod(tensor.shape)
                nonzero += nz_count
                total += total_params
                # print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, ({100 * nonzero / total:6.2f}% remained)')
        return (round((nonzero/total)*100, 1))

    
    # ======================================
    # =              Training              =
    # ======================================
    
    test_acc_list, nact_list = [], []

    for idx_epoch in range(parser_args.epochs):  # in total will run total_iter # of iterations, so total_epoch is not accurate
        # test
        test_acc = test(model, device, test_loader)
        test_acc_list.append(test_acc)
        nact = print_nonzeros(model, mask)
        nact_list.append(nact)
        result_df = pd.DataFrame({'test': test_acc_list, 'nact': nact_list})
        # write here
        result_df.to_csv("{}/LTH_cifar10_resnet20_round_{}_reinit_{}_shuffle_{}_chg_weight_{}_chg_mask_{}.csv".format(parser_args.dest_dir, parser_args.resume_round, reinit, shuffle, chg_weight, chg_mask), index=False)
        if idx_epoch >= parser_args.epochs - 1:  # don't need to train the last epoch
            PATH_model = "{}/model_round_{}_reinit_{}_shuffle_{}_chg_weight_{}_chg_mask_{}.pth".format(parser_args.dest_dir, parser_args.resume_round, reinit, shuffle, chg_weight, chg_mask)
            # write here
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, PATH_model)
            return

        # Training
        model.train()
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                model.train()
                imgs, targets = imgs.to(device), targets.to(device)
                output = model(imgs)
                train_loss = criterion(output, targets)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # mute the neurons again: because of numerical issue in pytorch
            for name, param in model.named_parameters():
                if name in model.prunable_layer_names:
                    tensor = param.data.detach()
                    param.data = tensor * mask[()][name].to(device).float()

        print('Train Epoch: {}/{} Loss: {:.4f} Test Acc: {:.2f}'.format(idx_epoch, parser_args.epochs, train_loss.item(), test_acc))
        if scheduler is not None:
            scheduler.step()

    return
                    


def main():
    global parser_args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer option to use |sgd|adam|')
    
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 14)')
    
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
        
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    
    parser.add_argument('--data', default='../data')
    parser.add_argument('--bias', action='store_true', default=False)

    # added parser for cifar 10 resnet18
    parser.add_argument("--ignore_name", default=None, help="the name of layers to be ignored. Example: 'linear1.weight, between.weight'.")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--gamma", default=0.1, type=float, help="learning rate scheduler")
    parser.add_argument("--resume-round", required=True, type=int, help="which round to resume to")
    parser.add_argument("--dest-dir", required=True)
    parser.add_argument("--rewind-model", required=True, default="short_imp/Liu_checkpoint_model_correct.pth")
    parser.add_argument("--gpu", required=True)

    parser_args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(parser_args.gpu) if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    set_seed(parser_args.seed)

    train_loader, test_loader = data_loader(parser_args)

    sanity_check(parser_args, train_loader, test_loader, device, shuffle=False, reinit=False, chg_mask=False, chg_weight=False)
    # sanity_check(parser_args, train_loader, test_loader, device, shuffle=False, reinit=True, chg_mask=False, chg_weight=False)
    # sanity_check(parser_args, train_loader, test_loader, device, shuffle=True, reinit=False, chg_mask=True, chg_weight=False)
    # sanity_check(parser_args, train_loader, test_loader, device, shuffle=True, reinit=False, chg_mask=False, chg_weight=True)



if __name__ == '__main__':
    main()


