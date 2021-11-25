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
    torch.backends.cudnn.benchmark = False # True
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


def IMP_train(parser_args, train_loader, test_loader, device):
    """
    :param parser_args
    :param train_loader, test_loader
    :param device
    """

    # ======================================
    # =           Initialization           =
    # ======================================
   
    use_amp = True

    assert parser_args.rewind_iter // 391 < parser_args.iter_period

    # weight initialization
    model = resnet20(ignore_name=parser_args.ignore_name).to(device)
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
    init_params(model)

    n_round = parser_args.epochs // parser_args.iter_period  # number of round (number of pruning happens)
    n_epoch = parser_args.iter_period  # number of epoch per round
    
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
        else:
            print("INVALID OPTIMIZER")
            print("EXITING")
            exit()
        return optimizer, scheduler


    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dest_dir = parser_args.dest_dir # "model/"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)


    # ======================================
    # =        Pre-define Function         =
    # ======================================

    def prune_by_percentile_global(percent, model, rewind_state_dict, current_mask):
        """This function prune the model by percentile
        :param percent: percentage of pruning for each round
        :param mask: the pervious mask
        :param model: the model to be pruned
        :param rewind_state_dict: rewind back to the state dict
        :returns: 
            model: the updated model (already rewind back to the rewind state, with new mask applied)
            mask: the updated mask
        """
        # ========= Prune =========
        current_mask = Mask.ones_like(model).numpy() if current_mask is None else current_mask.numpy()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(percent / 100. * number_of_remaining_weights).astype(int)

        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in model.state_dict().items()
                   if k in model.prunable_layer_names}  # get all the weights except the final linear layer

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in weights.items()})

        # ========= Update =========
        for name, weight in model.named_parameters():
            if name in model.prunable_layer_names:
                weight.data = new_mask[name].to(device) * rewind_state_dict[name].data
                # print(new_mask[name].sum(), new_mask[name].shape)
        # print("update weight")

        return model, new_mask

    
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
    
    test_acc_list, n_act_list = [], []
    before_acc_list = []
    mask = None
    counter = 0
    for idx_round in range(n_round):
        if idx_round > 0:
            model, mask = prune_by_percentile_global(parser_args.prune_perct, model, rewind_state_dict, mask)
            
        before_acc = test(model, device, test_loader)
        optimizer, scheduler = get_optimizer_and_scheduler(parser_args)
        print(f"\n--- Pruning Level [{idx_round}/{n_round}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = print_nonzeros(model, mask)

        # save the model
        PATH_model = os.path.join(dest_dir, "round_{}_model.pth".format(idx_round))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, PATH_model)
        if mask is not None:
            PATH_mask = os.path.join(dest_dir, "round_{}_mask.npy".format(idx_round))
            np.save(PATH_mask, mask, allow_pickle=True)

        if idx_round == 1:
            n_epoch = 150  # NOTE: hard code, change later
            
        for idx_epoch in range(n_epoch):  # in total will run total_iter # of iterations, so total_epoch is not accurate
            # Training
            model.train()
            for batch_idx, (imgs, targets) in enumerate(train_loader):
                counter += 1
                with torch.cuda.amp.autocast(enabled=use_amp):
                    model.train()
                    imgs, targets = imgs.to(device), targets.to(device)
                    output = model(imgs)
                    train_loss = criterion(output, targets)
                    # optimizer.zero_grad()
                    # train_loss.backward()
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # print(batch_idx, train_loss.item())
                optimizer.zero_grad()
                # train_loss.backward()

                # optimizer.step()
                # mute the neurons again: because of numerical issue in pytorch
                if idx_round != 0:
                    for name, param in model.named_parameters():
                        if name in model.prunable_layer_names:
                            tensor = param.data.detach()
                            param.data = tensor * mask[name].to(device).float()
                if idx_round == 0 and counter == parser_args.rewind_iter:
                    rewind_state_dict = copy.deepcopy(model.state_dict())
                    PATH_model = os.path.join(dest_dir, "Liu_checkpoint_model_correct.pth".format(idx_round))
                    torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                    }, PATH_model)

            test_acc = test(model, device, test_loader)
            test_acc_list.append(test_acc)
            before_acc_list.append(before_acc)
            n_act_list.append(comp1)
            print('Train Epoch: {}/{} Loss: {:.4f} Test Acc: {:.2f}'.format(idx_epoch, n_epoch, train_loss.item(), test_acc))
            if scheduler is not None:
                scheduler.step()
            # print("lr {}".format(optimizer.param_groups[0]["lr"]))
            
            result_df = pd.DataFrame({'test': test_acc_list, 'nact': n_act_list, "before": before_acc_list})
            result_df.to_csv("{}/LTH_cifar10_resnet20.csv".format(dest_dir), index=False)
        # save the model
        PATH_model_after = os.path.join(dest_dir, "round_{}_finish_model.pth".format(idx_round))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, PATH_model_after)
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
    
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    
    parser.add_argument('--iter_period', type=int, default=20,
                        help='period [epochs] for iterative pruning ')
    
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    
    parser.add_argument('--prune_perct', type=float, default=20, metavar='S',
                        help='prune percent for each layer')

    parser.add_argument('--data', default='../data')
    parser.add_argument('--bias', action='store_true', default=False)

    # added parser for cifar 10 resnet18
    parser.add_argument("--ignore_name", default=None, help="the name of layers to be ignored. Example: 'linear1.weight, between.weight'.")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--gamma", default=0.1, type=float, help="learning rate scheduler")
    parser.add_argument("--rewind_iter", default=1000, type=int, help="which epoch to rewind to")
    parser.add_argument('--dest_dir', default='model/')
    parser.add_argument("--gpu", default=1)


    parser_args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(parser_args.gpu) if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    set_seed(parser_args.seed)

    train_loader, test_loader = data_loader(parser_args)

    IMP_train(parser_args, train_loader, test_loader, device)



if __name__ == '__main__':
    main()


