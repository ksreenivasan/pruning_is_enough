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

# from cifar_model_resnet import resnet20
# from IMP_codebase.mask import Mask
from imp_mask import Mask

# for merge the code into parent directory
from args_helper import parser_args
from main_utils import get_model, get_dataset, get_optimizer, switch_to_wt
from utils.utils import set_seed
from utils.schedulers import get_scheduler


def redraw(model, mask, mask_bias, device, shuffle=False, reinit=False, chg_mask=False, chg_weight=False):
    cp_model = copy.deepcopy(model)
    cp_mask = copy.deepcopy(mask)
    if parser_args.bias:
        cp_mask_bias = copy.deepcopy(mask_bias)
    for name, param in cp_model.named_parameters():
        if name in cp_model.prunable_layer_names:
            weight = param.data.detach()
            if shuffle:
                if chg_mask:
                    try:  # will handle the case where mask is None
                        tmp_mask = cp_mask[()][name]
                        idx = torch.randperm(tmp_mask.nelement())
                        tmp_mask = tmp_mask.view(-1)[idx].view(tmp_mask.size())
                        mask[()][name] = tmp_mask
                    except:
                        pass
                if chg_weight:
                    idx = torch.randperm(weight.nelement())
                    weight = weight.view(-1)[idx].view(weight.size())
            elif reinit:
                init.kaiming_normal_(weight)
            elif not (shuffle and reinit):
                pass
            else:
                raise NotImplementedError
            if parser_args.imp_resume_round > 0:
                param.data = weight * mask[()][name].to(device).float()
            else:
                param.data = weight
    if parser_args.bias:
        for name, param in cp_model.named_parameters():
            if name in cp_model.prunable_biases:
                weight = param.data.detach()
                if shuffle:
                    if chg_mask:
                        try:  # will handle the case where mask is None
                            tmp_mask = cp_mask_bias[()][name]
                            idx = torch.randperm(tmp_mask.nelement())
                            tmp_mask = tmp_mask.view(-1)[idx].view(tmp_mask.size())
                            mask[()][name] = tmp_mask
                        except:
                            pass
                    if chg_weight:
                        idx = torch.randperm(weight.nelement())
                        weight = weight.view(-1)[idx].view(weight.size())
                elif reinit:
                    init.kaiming_normal_(weight)
                elif not (shuffle and reinit):
                    pass
                else:
                    raise NotImplementedError
                if parser_args.imp_resume_round > 0:
                    param.data = weight * mask[()][name].to(device).float()
                else:
                    param.data = weight
    return cp_model, mask


def sanity_check(parser_args, data, device, shuffle=False, reinit=False, chg_mask=False, chg_weight=False):
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
    dest_dir = os.path.join("results", parser_args.subfolder)

    # load model
    model = get_model(parser_args)
    model = switch_to_wt(model).to(device)
    if parser_args.imp_no_rewind:
        # then I will load the model right after being pruned
        PATH_model = os.path.join(dest_dir, "round_{}_model.pth".format(parser_args.imp_resume_round))
    else:
        PATH_model = parser_args.imp_rewind_model
    checkpoint = torch.load(PATH_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    # load mask
    if parser_args.imp_resume_round > 0:
        PATH_mask = "results/{}/round_{}_mask.npy".format(parser_args.subfolder, parser_args.imp_resume_round)
        mask = np.load(PATH_mask, allow_pickle=True)
        if parser_args.bias:
            PATH_mask_bias = os.path.join(dest_dir, "round_{}_mask_bias.npy".format(parser_args.imp_resume_round))
            mask_bias = np.load(PATH_mask_bias, allow_pickle=True)
        else:
            mask_bias = None
    else:
        mask, mask_bias = Mask.ones_like(model)

    # make change to mask or model
    model, mask = redraw(model, mask, mask_bias, device, shuffle, reinit, chg_mask, chg_weight)

    # Optimizer and criterion
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(parser_args, model)
    scheduler = get_scheduler(optimizer, parser_args.lr_policy, milestones=[80, 120], gamma=parser_args.lr_gamma) 
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

        test_acc = 100. * correct / len(test_loader.dataset)
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
        test_acc = test(model, device, data.val_loader)
        test_acc_list.append(test_acc)
        nact = print_nonzeros(model, mask)
        nact_list.append(nact)
        result_df = pd.DataFrame({'test': test_acc_list, 'nact': nact_list})
        # write here
        result_df.to_csv("results/{}/LTH_cifar10_resnet20_round_{}_reinit_{}_shuffle_{}_chg_weight_{}_chg_mask_{}.csv".format(parser_args.subfolder, parser_args.imp_resume_round, reinit, shuffle, chg_weight, chg_mask), index=False)
        if idx_epoch >= parser_args.epochs - 1:  # don't need to train the last epoch
            PATH_model = "results/{}/model_round_{}_reinit_{}_shuffle_{}_chg_weight_{}_chg_mask_{}.pth".format(parser_args.subfolder, parser_args.imp_resume_round, reinit, shuffle, chg_weight, chg_mask)
            # write here
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, PATH_model)
            return

        # Training
        model.train()
        for batch_idx, (imgs, targets) in enumerate(data.train_loader):
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
                    if parser_args.imp_resume_round > 0:
                        param.data = tensor * mask[()][name].to(device).float()
                        

        test_acc = test(model, device, data.val_loader)
        print('Train Epoch: {}/{} Loss: {:.4f} Test Acc: {:.2f}'.format(idx_epoch, parser_args.epochs, train_loss.item(), test_acc))
        if scheduler is not None:
            scheduler.step()

    return
                    


def main():
    global parser_args
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(parser_args.gpu) if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    set_seed(parser_args.seed)

    # train_loader, test_loader = data_loader(parser_args)
    data = get_dataset(parser_args)

    # finetune: all False
    sanity_check(parser_args, data, device, shuffle=False, reinit=False, chg_mask=False, chg_weight=False)
    if not parser_args.imp_no_rewind:
        # reinit
        sanity_check(parser_args, data, device, shuffle=False, reinit=True, chg_mask=False, chg_weight=False)
        # shuffle mask
        sanity_check(parser_args, data, device, shuffle=True, reinit=False, chg_mask=True, chg_weight=False)
        # shuffle weights
        sanity_check(parser_args, data, device, shuffle=True, reinit=False, chg_mask=False, chg_weight=True)



if __name__ == '__main__':
    main()


