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
from main_utils import get_model, get_dataset, get_optimizer, switch_to_wt, set_gpu
from utils.utils import set_seed
from utils.schedulers import get_scheduler

if parser_args.arch in ['transformer']:
    from transformer_main_utils import print_nonzeros as print_nonzeros_transformer
    from transformer_main_utils import train as train_transformer
    from transformer_main_utils import batchify, evaluate
    import transformer_data 
    import transformer_model
    from utils.builder import get_builder
    import time
    import math


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
    if parser_args.arch in ['transformer']:
        corpus = transformer_data.Corpus(parser_args.data)
        ntokens = len(corpus.dictionary)
        model = transformer_model.TransformerModel(get_builder(), ntokens, parser_args.transformer_emsize, parser_args.transformer_nhead, parser_args.transformer_nhid, parser_args.transformer_nlayers, parser_args.transformer_dropout).to(device)
        model = set_gpu(parser_args, model)
        criterion = nn.NLLLoss()
        eval_batch_size = 10
        train_data = batchify(corpus.train, parser_args.batch_size, device)
        val_data = batchify(corpus.valid, eval_batch_size, device)
        test_data = batchify(corpus.test, eval_batch_size, device)
        best_val_loss = None
    else:
        model = get_model(parser_args)
        criterion = nn.CrossEntropyLoss().to(device)

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
    if parser_args.imp_resume_round > 0:
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
        finish_index = parser_args.iter_period * parser_args.imp_resume_round
        test_acc_list = result_df["test"].tolist()[:finish_index]
        n_act_list = result_df["nact"].tolist()[:finish_index]
        before_acc_list = result_df["before"].tolist()[:finish_index]
    else:
        test_acc_list, n_act_list = [], []
        if parser_args.arch in ['transformer']:
            before_val_acc_list, before_test_acc_list, val_acc_list = [], [], []
        else:
            before_acc_list = []
        parser_args.imp_resume_round = 0
        mask, mask_bias = None, None

    n_round = parser_args.epochs // parser_args.iter_period  # number of round (number of pruning happens)
    n_epoch = parser_args.iter_period  # number of epoch per round
    print("{} round, each round takes {} epochs".format(n_round, n_epoch))

    # Optimizer and criterion
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(parser_args, model)
    # NOTE: hard code, just to make sure my code runs correctly
    # scheduler = get_scheduler(optimizer, parser_args.lr_policy, milestones=[80, 120], gamma=parser_args.lr_gamma)

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


    def put_mask_on(model, mask, mask_bias):
        for name, weight in model.named_parameters():
            if name in model.prunable_layer_names:
                weight.data = mask[name].to(device) * weight.data
            if parser_args.bias:
                if name in model.prunable_biases:
                    weight.data = mask_bias[name].to(device) * weight.data

        return model

    
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

    
    def print_nonzeros(model):
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
        if idx_round > parser_args.imp_resume_round:
            # model, mask = prune_by_percentile_global(parser_args.prune_perct, model, rewind_state_dict, mask)
            model, mask, mask_bias = prune_by_percentile_global(parser_args.prune_rate, model, rewind_state_dict, mask, mask_bias)
        elif parser_args.imp_resume_round > 0:
            assert mask is not None
            model = put_mask_on(model, mask, mask_bias)
        else:
            pass

        print(f"\n--- Pruning Level [{idx_round}/{n_round}]: ---")
        if parser_args.arch in ['transformer']:
            before_val_acc = evaluate(parser_args, model, ntokens, criterion, val_data)
            before_test_acc = evaluate(parser_args, model, ntokens, criterion, test_data)
            # comp1 = print_nonzeros_transformer(model)
            comp1 = print_nonzeros(model)
        else:
            before_acc = test(model, device, data.val_loader)
            comp1 = print_nonzeros(model)
        # optimizer, scheduler = get_optimizer_and_scheduler(parser_args)
        optimizer = get_optimizer(parser_args, model)
        # scheduler = get_scheduler(optimizer, parser_args.lr_policy, gamma=parser_args.lr_gamma)
        # NOTE: hard code
        if n_epoch in [6]:
            scheduler = get_scheduler(optimizer, parser_args.lr_policy, milestones=[3,], gamma=parser_args.lr_gamma)
        # elif n_epoch == 200: 
            # scheduler = get_scheduler(optimizer, parser_args.lr_policy, milestones=[100, 150], gamma=parser_args.lr_gamma)

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
            if parser_args.arch in ['transformer']:
                epoch_start_time = time.time()
                train_transformer(parser_args, idx_epoch, ntokens, train_data, model, optimizer, criterion, mask, mask_bias)
                if scheduler is not None:
                    scheduler.step()
                val_loss = evaluate(parser_args, model, ntokens, criterion, val_data)
                test_loss = evaluate(parser_args, model, ntokens, criterion, test_data)
                val_acc_list.append(val_loss)
                test_acc_list.append(test_loss)
                avg_sparsity = print_nonzeros(model)  # print_nonzeros_transformer(model)
                n_act_list.append(avg_sparsity)
                before_val_acc_list.append(before_val_acc)
                before_test_acc_list.append(before_test_acc)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(idx_epoch, (time.time() - epoch_start_time),
                                                   val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(os.path.join("results", parser_args.subfolder, "finetune_model.pt"), 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    pass
                    # for param_group in optimizer.param_groups:
                    #     param_group["lr"] /= 4.0

                result_df = pd.DataFrame({'val': val_acc_list, 'nact': n_act_list, "test": test_acc_list, 'before_val': before_val_acc_list, 'before_test': before_test_acc_list})

            else:
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

    if parser_args.arch in ['transformer']:
        data = None
    else:
        data = get_dataset(parser_args)

    IMP_train(parser_args, data, device)



if __name__ == '__main__':
    main()


