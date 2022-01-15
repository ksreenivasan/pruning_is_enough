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


def IMP_train(parser_args, data, device):
    """
    :param parser_args
    :param train_loader, test_loader
    :param device
    """

    # ======================================
    # =           Initialization           =
    # ======================================
   
    use_amp = True

    if not parser_args.imp_no_rewind:
        assert parser_args.imp_rewind_iter // 391 < parser_args.iter_period  # NOTE: hard code, needs to modify later
    dest_dir = os.path.join("results", parser_args.subfolder)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    # weight initialization
    model = get_model(parser_args)
    model = switch_to_wt(model).to(device)

    if parser_args.imp_no_rewind:
        rewind_state_dict = None
    elif parser_args.imp_rewind_iter == 0:  # handle the case where rewind to initial weights
        rewind_state_dict = copy.deepcopy(model.state_dict())
        PATH_model = os.path.join(dest_dir, "Liu_checkpoint_model_correct.pth")
        torch.save({
                    'model_state_dict': model.state_dict(),
        }, PATH_model)
    else:  # rewind to early training phase
        pass

    # resume at some point
    if parser_args.imp_resume_round != -1:
        ckpt = torch.load(os.path.join(dest_dir, "Liu_checkpoint_model_correct.pth"), map_location='cpu')
        model.load_state_dict(ckpt["model_state_dict"])
        rewind_state_dict = copy.deepcopy(model.state_dict())
        # load mask
        PATH_mask = "results/{}/round_{}_mask.npy".format(parser_args.subfolder, parser_args.imp_resume_round)
        mask = np.load(PATH_mask, allow_pickle=True)[()]
        if parser_args.bias:
            PATH_mask_bias = os.path.join(dest_dir, "round_{}_mask_bias.npy".format(parser_args.imp_resume_round))
            mask_bias = np.load(PATH_mask_bias, allow_pickle=True)[()]
        else:
            mask_bias = None
        # load csv file
        result_df = pd.read_csv(os.path.join(dest_dir, "LTH_cifar10_resnet20.csv"))
        test_acc_list = result_df["test"]
        n_act_list = result_df["nact"]
        before_acc_list = result_df["before"]
    else:
        test_acc_list, n_act_list, before_acc_list = [], [], []
        parser_args.imp_resume_round = 0
        mask, mask_bias = None, None

    n_round = parser_args.epochs // parser_args.iter_period  # number of round (number of pruning happens)
    n_epoch = parser_args.iter_period  # number of epoch per round
    
    # Optimizer and criterion
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(parser_args, model)
    # NOTE: hard code, just to make sure my code runs correctly
    scheduler = get_scheduler(optimizer, parser_args.lr_policy, milestones=[80, 120], gamma=parser_args.lr_gamma)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


    # ======================================
    # =        Pre-define Function         =
    # ======================================

    def prune_by_percentile_global(percent, model, rewind_state_dict, current_mask, current_mask_bias):
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
        if current_mask is None:
            # if parser_args.bias == True, current_mask_bias is not None; otherwise it is None.
            current_mask, current_mask_bias = Mask.ones_like(model)
        current_mask = current_mask.numpy()
        if parser_args.bias:
            current_mask_bias = current_mask_bias.numpy()
        
        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(percent * number_of_remaining_weights).astype(int)
        if parser_args.bias:  # NOTE: bias use the same pruning rate as weight
            number_of_remaining_bias = np.sum([np.sum(v) for v in current_mask_bias.values()])
            number_of_bias_to_prune = np.ceil(percent * number_of_remaining_bias).astype(int)

        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in model.state_dict().items()
                   if k in model.prunable_layer_names}
        if parser_args.bias:
            bias = {k: v.clone().cpu().detach().numpy()
                    for k, v in model.state_dict().items()
                    if k in model.prunable_biases}

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]
        new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in weights.items()})
        if parser_args.bias:
            bias_vector = np.concatenate([v[current_mask_bias[k] == 1] for k, v in bias.items()])
            bias_threshold = np.sort(np.abs(bias_vector))[number_of_bias_to_prune]
            new_mask_bias = Mask({k: np.where(np.abs(v) > bias_threshold, current_mask_bias[k], np.zeros_like(v))
                         for k, v in bias.items()})
        else:
            new_mask_bias = None

        # ========= Update =========
        for name, weight in model.named_parameters():
            if name in model.prunable_layer_names:
                if parser_args.imp_no_rewind:
                    weight.data = new_mask[name].to(device) * weight.data
                else:
                    weight.data = new_mask[name].to(device) * rewind_state_dict[name].data
            if parser_args.bias:
                if name in model.prunable_biases:
                    if parser_args.imp_no_rewind:
                        weight.data = new_mask_bias[name].to(device) * weight.data
                    else:
                        weight.data = new_mask_bias[name].to(device) * rewind_state_dict[name].data

        return model, new_mask, new_mask_bias

    
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
            if name in model.prunable_layer_names or name in model.prunable_biases:
                tensor = p.data.detach().cpu().numpy()
                nz_count = np.count_nonzero(tensor)
                total_params = np.prod(tensor.shape)
                nonzero += nz_count
                total += total_params
                print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, ({100 * nonzero / total:6.2f}% remained)')
        return (round((nonzero/total)*100, 1))

    
    # ======================================
    # =              Training              =
    # ======================================
    
    # test_acc_list, n_act_list = [], []
    # before_acc_list = []
    counter = 0
    for idx_round in range(parser_args.imp_resume_round, n_round):
        if idx_round > 0:
            # model, mask = prune_by_percentile_global(parser_args.prune_perct, model, rewind_state_dict, mask)
            model, mask, mask_bias = prune_by_percentile_global(parser_args.prune_rate, model, rewind_state_dict, mask, mask_bias)
        before_acc = test(model, device, data.val_loader)
        # optimizer, scheduler = get_optimizer_and_scheduler(parser_args)
        optimizer = get_optimizer(parser_args, model)
        # NOTE: hard code
        if n_epoch in [150, 160]:
            scheduler = get_scheduler(optimizer, parser_args.lr_policy, milestones=[80, 120], gamma=parser_args.lr_gamma)
        elif n_epoch == 200: 
            scheduler = get_scheduler(optimizer, parser_args.lr_policy, milestones=[100, 150], gamma=parser_args.lr_gamma)

        print(f"\n--- Pruning Level [{idx_round}/{n_round}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = print_nonzeros(model, mask)

        # save the model and mask right after prune
        PATH_model = os.path.join(dest_dir, "round_{}_model.pth".format(idx_round))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, PATH_model)
        if mask is not None:
            PATH_mask = os.path.join(dest_dir, "round_{}_mask.npy".format(idx_round))
            np.save(PATH_mask, mask, allow_pickle=True)

            if parser_args.bias:
                PATH_mask_bias = os.path.join(dest_dir, "round_{}_mask_bias.npy".format(idx_round))
                np.save(PATH_mask_bias, mask_bias, allow_pickle=True)

        
        for idx_epoch in range(n_epoch):  # in total will run total_iter # of iterations, so total_epoch is not accurate
            # Training
            model.train()
            for batch_idx, (imgs, targets) in enumerate(data.train_loader):
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
                optimizer.zero_grad()
                # train_loss.backward()

                # optimizer.step()
                # mute the neurons again: because of numerical issue in pytorch
                if idx_round != 0:
                    for name, param in model.named_parameters():
                        if name in model.prunable_layer_names:
                            tensor = param.data.detach()
                            param.data = tensor * mask[name].to(device).float()
                if idx_round == 0 and counter == parser_args.imp_rewind_iter and (not parser_args.imp_no_rewind):
                    PATH_model = os.path.join(dest_dir, "Liu_checkpoint_model_correct.pth")
                    assert counter > 0
                    rewind_state_dict = copy.deepcopy(model.state_dict())                    
                    torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                    }, PATH_model)

            test_acc = test(model, device, data.val_loader)
            test_acc_list.append(test_acc)
            before_acc_list.append(before_acc)
            n_act_list.append(comp1)
            print('Train Epoch: {}/{} Loss: {:.4f} Test Acc: {:.2f}'.format(idx_epoch, n_epoch, train_loss.item(), test_acc))
            if scheduler is not None:
                scheduler.step()
            
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
    # use the parser_args from args_helper.py
    global parser_args
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(parser_args.gpu) if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    set_seed(parser_args.seed)

    data = get_dataset(parser_args)

    IMP_train(parser_args, data, device)



if __name__ == '__main__':
    main()


