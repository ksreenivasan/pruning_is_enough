"""
### main_utils.py
### put every long functions in main.py into here
"""

from args_helper import parser_args
import pdb
import numpy as np
import os
import pathlib
import random
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp

import sys
import re

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    round_model,
    get_model_sparsity,
    prune,
    redraw,
    get_layers,
    get_prune_rate,
)
from utils.schedulers import get_scheduler
from utils.utils import set_seed, plot_histogram_scores
from SmartRatio import SmartRatio

import importlib

import data
import models

import copy
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors



def print_layers(parser_args, model):
    conv_layers, linear_layers = get_layers(parser_args.arch, model)
    i = 0
    for layer in [*conv_layers, *linear_layers]:
        i += 1
        print(i, layer)


def print_model(model, parser_args):
    #from torchsummary import summary
    #summary(model.cuda(), (3,32,32)) # for cifar
    # check the model architecture
    
    num_params = 0
    if parser_args.algo == 'training':
        for name, param in model.named_parameters():
            print(name, param.view(-1).numel())
            #pdb.set_trace()
            num_params += param.view(-1).numel()
    else:
        for name, param in model.named_parameters():
            if name.endswith('.scores'):
                print(name, param.view(-1).numel())
                num_params += param.view(-1).numel()

    '''
    else:
        conv_layers, linear_layers = get_layers(parser_args.arch, model)
        for layer in [*conv_layers, *linear_layers]:
            print(layer, layer.scores.view(-1).shape)
    '''
    print('total num_params: ', num_params)
    #exit()


def do_sanity_checks(model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list, val_acc_list,
                     reg_loss_list, model_sparsity_list, result_root):

    print("Beginning Sanity Checks:")
    # do the sanity check for shuffled mask/weights, reinit weights
    print("Sanity Check 1: Weight Reinit")
    cp_model = copy.deepcopy(model)
    cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list, val_acc_list,
                        reg_loss_list, model_sparsity_list, result_root, reinit=True, chg_weight=True)

    '''
    print("Sanity Check 2: Weight Reshuffle")
    cp_model = copy.deepcopy(model)
    cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list,
                        reg_loss_list, model_sparsity_list, result_root, shuffle=True, chg_weight=True)
    '''
    print("Sanity Check 2: Mask Reshuffle")
    cp_model = copy.deepcopy(model)
    cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list, val_acc_list,
                        reg_loss_list, model_sparsity_list, result_root, shuffle=True, chg_mask=True)

    # this doesn't work. removing it.
    """
    print("Sanity Check 3: Mask Invert")
    cp_model = copy.deepcopy(model)
    cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list,
                        reg_loss_list, model_sparsity_list, result_root, invert=True, chg_mask=True)
    """


def save_checkpoint_at_prune(model, parser_args):
    # let's see if we can get all sparsity plots with one run
    # save checkpoints at every pruned model so that we can finetune later
    # save checkpoint for later debug
    cp_model = round_model(model, parser_args.round, noise=parser_args.noise,
                           ratio=parser_args.noise_ratio, rank=parser_args.gpu)
    avg_sparsity = get_model_sparsity(cp_model)
    idty_str = get_idty_str(parser_args)
    if not os.path.isdir('model_checkpoints/'):
        os.mkdir('model_checkpoints/')
    ckpt_root = 'model_checkpoints/ckpts_' + idty_str + '/'
    if not os.path.isdir(ckpt_root):
        os.mkdir(ckpt_root)
    model_filename = ckpt_root + \
        "hc_ckpt_at_sparsity_{}.pt".format(int(avg_sparsity))
    print("Checkpointing model to {}".format(model_filename))
    torch.save(model.state_dict(), model_filename)


def evaluate_without_training(parser_args, model, model2, validate, data, criterion):
    if parser_args.algo in ['hc_iter']:
        model = round_model(model, parser_args.round, noise=parser_args.noise,
                            ratio=parser_args.noise_ratio, rank=parser_args.gpu)
        eval_and_print(validate, data.val_loader, model, criterion, parser_args, writer=None,
                       epoch=parser_args.start_epoch, description='final model after rounding')
    for trial in range(parser_args.num_test):
        if parser_args.algo in ['hc']:
            if parser_args.how_to_connect == "prob":
                cp_model = round_model(model, parser_args.round, noise=parser_args.noise,
                                       ratio=parser_args.noise_ratio, rank=parser_args.gpu)
            else:
                cp_model = copy.deepcopy(model)
            eval_and_print(validate, data.val_loader, cp_model, criterion, parser_args,
                           writer=None, epoch=parser_args.start_epoch, description='model after pruning')
    if parser_args.pretrained2:
        eval_and_print(validate, data.val_loader, model2, criterion, parser_args,
                       writer=None, epoch=parser_args.start_epoch, description='model2')
        if parser_args.algo in ['hc']:
            if parser_args.how_to_connect == "prob":
                cp_model2 = round_model(model2, parser_args.round, noise=parser_args.noise,
                                        ratio=parser_args.noise_ratio, rank=parser_args.gpu)
            else:
                cp_model2 = copy.deepcopy(model2)
            eval_and_print(validate, data.val_loader, cp_model2, criterion, parser_args,
                           writer=None, epoch=parser_args.start_epoch, description='model2 after pruning')
    if parser_args.pretrained and parser_args.pretrained2 and parser_args.mode_connect:
        if parser_args.weight_training:
            print('We are connecting weights')
            connect_weight(cp_model, criterion, data, validate, cp_model2)
        elif parser_args.algo in ['hc', 'ep', 'global_ep']:
            print('We are connecting masks')
            connect_mask(cp_model, criterion, data, validate, cp_model2)
        # visualize_mask_2D(cp_model, criterion, data, validate)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



#def test_smart_ratio(model, data, criterion, parser_args, result_root):
  


def test_random_subnet(model, data, criterion, parser_args, result_root, smart_ratio=-1):

    if smart_ratio != -1:
        # get a randomly pruned model with SmartRatio
        smart_ratio_args = {'linear_keep_ratio': 0.3, 
                            }
        smart_ratio_args = dotdict(smart_ratio_args)
        model = SmartRatio(model, smart_ratio_args, parser_args)
        # # NOTE: temporarily added for code checking
        # torch.save(model.state_dict(), result_root + 'init_model_{}.pth'.format(smart_ratio)) 
        # return
        model = set_gpu(parser_args, model)
        # this model modify `flag` to represent the sparsity,
        # and `score` are all ones.

    else:
        # round the score (in the model itself)
        model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)    
        
        # TODO: CHANGE THIS BACK once the finetune from checkpoints code is fixed
        # NOTE: this part is hard coded
        model = redraw(model, shuffle=parser_args.shuffle, reinit=parser_args.reinit, chg_mask=parser_args.chg_mask, chg_weight=parser_args.chg_weight)  

    model_filename = result_root + 'model_before_finetune.pth'
    print("Writing init model to {}".format(model_filename))
    torch.save(model.state_dict(), model_filename)

    old_epoch_list, old_test_acc_before_round_list, old_test_acc_list, old_reg_loss_list, old_model_sparsity_list = [], [], [], [], []
    model = finetune(model, parser_args, data, criterion, old_epoch_list, old_test_acc_before_round_list, old_test_acc_list, old_reg_loss_list, old_model_sparsity_list, result_root, shuffle=False, reinit=False, invert=False, chg_mask=False, chg_weight=False)

    # save checkpoint for later debug
    model_filename = result_root +  'model_after_finetune.pth'
    print("Writing final model to {}".format(model_filename))
    torch.save(model.state_dict(), model_filename)


def eval_and_print(validate, data_loader, model, criterion, parser_args, writer=None, epoch=parser_args.start_epoch, description='model'):

    acc1, acc5, acc10 = validate(
        data_loader, model, criterion, parser_args, writer=None, epoch=parser_args.start_epoch)
    print('Performance of {}'.format(description))
    print('acc1: {}, acc5: {}, acc10: {}'.format(acc1, acc5, acc10))

    return acc1


def finetune(model, parser_args, data, criterion, old_epoch_list, old_test_acc_before_round_list, old_test_acc_list, old_val_acc_list, old_reg_loss_list, old_model_sparsity_list, result_root, shuffle=False, reinit=False, invert=False, chg_mask=False, chg_weight=False):
    epoch_list = copy.deepcopy(old_epoch_list)
    test_acc_before_round_list = copy.deepcopy(old_test_acc_before_round_list)
    test_acc_list = copy.deepcopy(old_test_acc_list)
    val_acc_list = copy.deepcopy(old_val_acc_list)
    reg_loss_list = copy.deepcopy(old_reg_loss_list)
    model_sparsity_list = copy.deepcopy(old_model_sparsity_list)

    if parser_args.results_filename:
        result_root = parser_args.results_filename + '_'

    if parser_args.bottom_k_on_forward:
        prune(model, update_scores=True)
    elif parser_args.algo in ['hc', 'hc_iter']:
        if parser_args.unflag_before_finetune:
            # want to ensure that all weights are available to train, except for those that have been pruned
            model = round_model(model, round_scheme="all_ones", noise=parser_args.noise,
                                ratio=parser_args.noise_ratio, rank=parser_args.gpu)
            # check sparsity
            post_round_sparsity = get_model_sparsity(model)
        else:
            # round the score (in the model itself)
            model = round_model(model, round_scheme=parser_args.round, noise=parser_args.noise,
                                ratio=parser_args.noise_ratio, rank=parser_args.gpu)
            post_round_sparsity = get_model_sparsity(model)
    elif parser_args.algo in ['ep']:
        post_round_sparsity = get_model_sparsity(model)

    # apply reinit/shuffling masks/weights (if necessary)
    model = redraw(model, shuffle=shuffle, reinit=reinit,
                   invert=invert, chg_mask=chg_mask, chg_weight=chg_weight)

    # switch to weight training mode (turn on the requires_grad for weight/bias, and turn off the requires_grad for other parameters)
    model = switch_to_wt(model)

    # not to use score regulaization during the weight training
    parser_args.regularization = False

    # set base_setting and evaluate
    run_base_dir, ckpt_base_dir, log_base_dir, writer, epoch_time, validation_time, train_time, progress_overall = get_settings(
        parser_args)

    optimizer = get_optimizer(parser_args, model, finetune_flag=True)
    scheduler = get_scheduler(optimizer, policy=parser_args.fine_tune_lr_policy)
    ''' 
    if parser_args.epochs == 150:
        scheduler = get_scheduler(optimizer, parser_args.fine_tune_lr_policy, milestones=[
                                  80, 120], gamma=0.1)  # NOTE: hard-coded
    elif parser_args.epochs == 50:
        scheduler = get_scheduler(optimizer, parser_args.fine_tune_lr_policy, milestones=[
                                  20, 40], gamma=0.1)  # NOTE: hard-coded
    elif parser_args.epochs == 200:
        scheduler = get_scheduler(optimizer, parser_args.fine_tune_lr_policy, milestones=[
                                  100, 150], gamma=0.1)  # NOTE: hard-coded
    elif parser_args.epochs == 300:
        scheduler = get_scheduler(optimizer, parser_args.fine_tune_lr_policy, milestones=[
                                  150, 250], gamma=0.1)  # NOTE: hard-coded
    else:
        scheduler = get_scheduler(optimizer, parser_args.fine_tune_lr_policy, milestones=[
                                  20, 40], gamma=0.1)  # NOTE: hard-coded
    '''

    train, validate, modifier = get_trainer(parser_args)

    # check the performance of loaded model (after rounding)
    acc1, acc5, acc10 = validate(
        data.val_loader, model, criterion, parser_args, writer, parser_args.epochs-1)
    val_acc1, val_acc5, val_acc10 = validate(
        data.actual_val_loader, model, criterion, parser_args, writer, parser_args.epochs-1)
    avg_sparsity = post_round_sparsity
    epoch_list.append(parser_args.epochs-1)
    test_acc_before_round_list.append(-1)
    test_acc_list.append(acc1)
    val_acc_list.append(val_acc1)
    reg_loss_list.append(0.0)
    model_sparsity_list.append(avg_sparsity)

    end_epoch = time.time()
    for epoch in range(parser_args.epochs, parser_args.epochs*2):

        if parser_args.multiprocessing_distributed:
            data.train_loader.sampler.set_epoch(epoch)
        # lr_policy(epoch, iteration=None)
        # modifier(parser_args, epoch, model)
        cur_lr = get_lr(optimizer)
        print('epoch: {}, lr: {}'.format(epoch, cur_lr))

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5, train_acc10, reg_loss = train(
            data.train_loader, model, criterion, optimizer, epoch, parser_args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5, acc10 = validate(
            data.val_loader, model, criterion, parser_args, writer, epoch)
        val_acc1, val_acc5, val_acc10 = validate(
            data.actual_val_loader, model, criterion, parser_args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)
        # copy & paste the sparsity of prev. epoch
        avg_sparsity = model_sparsity_list[-1]

        # update all results lists
        epoch_list.append(epoch)
        test_acc_before_round_list.append(-1)
        test_acc_list.append(acc1)
        val_acc_list.append(val_acc1)
        reg_loss_list.append(reg_loss)
        model_sparsity_list.append(avg_sparsity)

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )
        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

        results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc_before_rounding': test_acc_before_round_list, 'test_acc': test_acc_list, 'val_acc': val_acc_list,
                                   'regularization_loss': reg_loss_list, 'model_sparsity': model_sparsity_list})
        if not chg_mask and not chg_weight:
            results_filename = result_root + 'acc_and_sparsity.csv'
        # elif chg_weight and shuffle:
        #    results_filename = result_root + 'acc_and_sparsity_weight_shuffle.csv'
        elif chg_mask and shuffle:
            results_filename = result_root + 'acc_and_sparsity_mask_shuffle.csv'
        elif chg_mask and invert:
            results_filename = result_root + 'acc_and_sparsity_mask_invert.csv'
        elif chg_weight and reinit:
            results_filename = result_root + 'acc_and_sparsity_weight_reinit.csv'
        else:
            raise NotImplementedError

        print("Writing results into: {}".format(results_filename))
        results_df.to_csv(results_filename, index=False)
        scheduler.step()

    return model


def get_idty_str(parser_args):
    train_mode_str = 'weight_training' if parser_args.weight_training else 'pruning'
    dataset_str = parser_args.dataset
    model_str = parser_args.arch
    algo_str = parser_args.algo
    rate_str = parser_args.prune_rate
    period_str = parser_args.iter_period
    reg_str = 'reg_{}'.format(parser_args.regularization)
    reg_lmbda = parser_args.lmbda if parser_args.regularization else ''
    opt_str = parser_args.optimizer
    policy_str = parser_args.lr_policy
    lr_str = parser_args.lr
    lr_gamma = parser_args.lr_gamma
    lr_adj = parser_args.lr_adjust
    finetune_lr_str = parser_args.fine_tune_lr
    fan_str = parser_args.scale_fan
    w_str = parser_args.init
    s_str = parser_args.score_init
    width_str = parser_args.width
    seed_str = parser_args.seed + parser_args.trial_num - 1
    run_idx_str = parser_args.run_idx
    lam_ft_str = parser_args.lam_finetune_loss
    n_step_ft_str = parser_args.num_step_finetune
    idty_str = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_finetune_{}_MAML_{}_{}_fan_{}_{}_{}_width_{}_seed_{}_idx_{}".\
        format(train_mode_str, dataset_str, model_str, algo_str, rate_str, period_str, reg_str, reg_lmbda,
               opt_str, policy_str, lr_str, lr_gamma, lr_adj, finetune_lr_str, lam_ft_str, n_step_ft_str, fan_str, w_str, s_str,
               width_str, seed_str, run_idx_str).replace(".", "_")

    return idty_str


def get_settings(parser_args):

    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(parser_args)
    parser_args.ckpt_base_dir = ckpt_base_dir
    writer = SummaryWriter(log_dir=log_base_dir)
    # writer = None
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    return run_base_dir, ckpt_base_dir, log_base_dir, writer, epoch_time, validation_time, train_time, progress_overall


def compare_rounding(validate, data_loader, model, criterion, parser_args, result_root):

    # generate supermask from naive rounding
    naive_model = round_model(model, 'naive')
    naive_mask, _ = get_mask(naive_model)

    idx_list, test_acc_list, dist_list, mask_list = [], [], [], []
    n_rand = 10
    for i in range(n_rand):
        # generate supermask from probabilistic rounding
        prob_model = round_model(model, 'prob')
        # prob_model = round_model(model, 'naive_prob')

        # evaluate and check the hamming distance btw naive_model & prob_model
        acc1 = eval_and_print(validate, data_loader, prob_model, criterion,
                              parser_args, description='probabilistic model {}'.format(i))
        idx_list.append(i)
        test_acc_list.append(acc1)

        prob_mask, _ = get_mask(prob_model)
        hamm_dist = torch.sum(
            torch.abs(naive_mask - prob_mask))/len(naive_mask)
        dist_list.append(hamm_dist.data.item())
        mask_list.append(prob_mask.data)

    # save the result in the dataframe
    compare_df = pd.DataFrame(
        {'idx': idx_list, 'test_acc': test_acc_list, 'hamming dist to naive': dist_list})
    results_filename = result_root + 'compare_rounding.csv'
    print("Writing rounding compare results into: {}".format(results_filename))
    compare_df.to_csv(results_filename, index=False)

    compare_prob = np.zeros((10, 10))
    for i in range(n_rand):
        for j in range(n_rand):
            compare_prob[i, j] = torch.sum(
                torch.abs(mask_list[i] - mask_list[j])/len(mask_list[i]))
    print(compare_prob)
    pd.DataFrame(compare_prob).to_csv(
        result_root + 'compare_probs.csv', header=None, index=False)

    return


# switches off gradients for scores and flags and switches it on for weights and biases
def switch_to_wt(model):
    print('Switching to weight training by switching off requires_grad for scores and switching it on for weights.')

    # this is for the case considering finetune loss
    parser_args.lam_finetune_loss = 0
    for name, params in model.named_parameters():
        # make sure param_name ends with .weight or .bias
        if re.match('.*\.weight', name):
            params.requires_grad = True
        elif parser_args.bias and re.match('.*\.bias$', name):
            params.requires_grad = True
        elif "score" in name:
            params.requires_grad = False
        else:
            # flags and everything else
            params.requires_grad = False

    return model


def get_mask(model):

    flat_tensor = []
    for name, params in model.named_parameters():
        if ".score" in name:
            flat_tensor.append(params.data)
            # print(name, params.data)
    # a: flat_tensor, b = mask_init,
    mask = _flatten_dense_tensors(flat_tensor)

    return mask, flat_tensor


def setup_distributed(ngpus_per_node):
    # for debugging
    #    os.environ['NCCL_DEBUG'] = 'INFO'
    #    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

    # setup environment
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'


def cleanup_distributed():
    torch.distributed.destroy_process_group()


# connect two masks trained by pruning
def connect_mask(model, criterion, data, validate, model2=None):
    # concatenate the masks
    flat_tensor = []
    flat_weight = []
    for name, params in model.named_parameters():
        if ".weight" in name:
            flat_weight.append(params.data)
        if ".score" in name:
            flat_tensor.append(params.data)
            # print(name, params.data)
    # a: flat_tensor, b = mask_init,
    mask_init = _flatten_dense_tensors(flat_tensor)

    flat_tensor2 = []
    idx = 0
    for name, params in model2.named_parameters():
        if ".weight" in name:
            print(name, torch.sum(torch.abs(flat_weight[idx] - params.data)))
            idx += 1
        if ".score" in name:
            flat_tensor2.append(params.data)
            print(name, params.data.shape)
    # a: flat_tensor2, b = mask_fin,
    mask_fin = _flatten_dense_tensors(flat_tensor2)

    # select random direction to go
    num_d = 1  # 100
    num_v = 1  # 5 # 100
    resol = 100  # 100  # 1000

    # batch data to test
    for data_, label_ in data.train_loader:
        data_, label_ = data_.cuda(), label_.cuda()
        break

    # setting for saving results
    cp_model = copy.deepcopy(model)
    dist_list = []
    train_mode_str = 'weight_training' if parser_args.weight_training else 'pruning'

    # init_time = time.time()
    for d1_idx in range(num_d):
        train_loss_mean_list = []
        train_loss_std_list = []
        train_acc_mean_list = []
        train_acc_std_list = []
        test_acc_mean_list = []
        test_acc_std_list = []
        # when 2nd model is not specified (use random direction)
        if model2 is None:
            sparsity1 = 0.2
            d1 = torch.bernoulli(torch.ones_like(mask_init) * sparsity1)  # d1
            # print('sum of d1: ', torch.sum(d1))
            new_d1 = (d1 + mask_init) % 2
        else:
            new_d1 = mask_fin
        normalized_hamming_dist = (
            torch.sum(torch.abs(mask_init - new_d1))/len(mask_init)).data.item()
        print('dist btw mask_src and mask_dest: ', normalized_hamming_dist)

        for i in range(resol+1):
            p = i/resol  # probability of sampling new_d1
            if d1_idx == 0:
                if model2 is None:
                    dist_list.append(round(p * sparsity1, 4))
                else:
                    dist_list.append(round(p * normalized_hamming_dist, 4))

            # loss_avg = 0
            # acc_avg = 0
            loss_arr, train_acc_arr, acc_arr = np.zeros(
                num_v), np.zeros(num_v), np.zeros(num_v)

            for v_idx in range(num_v):
                if parser_args.how_to_connect == "prob":
                    # [0, 1]^n  0 : I'll sample mask_init, 1: I'll sample d1
                    sampling_vct = torch.bernoulli(
                        torch.ones_like(mask_init) * p)
                    new_mask = mask_init * \
                        (1-sampling_vct) + new_d1 * sampling_vct   # w+v
                else:
                    new_mask = mask_init * p + mask_fin * (1-p)

                # pdb.set_trace()
                # print(torch.sum(torch.abs(new_mask - new_d1)))
                # put merged masks back to the model
                new_mask_unflat = _unflatten_dense_tensors(
                    new_mask, flat_tensor)
                idx = 0
                for name, params in cp_model.named_parameters():
                    if ".score" in name:
                        params.data = new_mask_unflat[idx]
                        # print(name, params.data.shape)
                        # print(torch.sum(torch.abs(params.data - flat_tensor2[idx])))
                        idx += 1

                if parser_args.how_to_connect == "round":
                    cp_model = round_model(cp_model, parser_args.round, noise=parser_args.noise,
                                           ratio=parser_args.noise_ratio, rank=parser_args.gpu)

                # compute loss for the mask
                loss = criterion(cp_model(data_), label_)
                acc1, acc5, acc10 = validate(
                    data.val_loader, cp_model, criterion, parser_args,
                    writer=None, epoch=parser_args.start_epoch)

                train_acc1, train_acc5, train_acc10 = validate(
                    data.train_loader, cp_model, criterion, parser_args,
                    writer=None, epoch=parser_args.start_epoch)

                print(i, v_idx, loss.data.item(), acc1, train_acc1)
                loss_arr[v_idx] = loss.data.item()
                acc_arr[v_idx] = acc1
                train_acc_arr[v_idx] = train_acc1

            train_loss_mean_list.append(np.mean(loss_arr))
            train_loss_std_list.append(np.std(loss_arr))
            train_acc_mean_list.append(np.mean(train_acc_arr))
            train_acc_std_list.append(np.std(train_acc_arr))
            test_acc_mean_list.append(np.mean(acc_arr))
            test_acc_std_list.append(np.std(acc_arr))

        if d1_idx == 0:
            results_df = pd.DataFrame({'dist': dist_list, 'train_loss_mean': train_loss_mean_list, 'train_loss_std': train_loss_std_list,
                                       'train_acc_mean': train_acc_mean_list, 'train_acc_std': train_acc_std_list,
                                       'test_acc_mean': test_acc_mean_list, 'test_acc_std': test_acc_std_list})
        else:
            raise NotImplementedError
            # results_df['batch_train_loss{}'.format(d1_idx+1)] = train_loss_list

        # fin_time = time.time()
        # print('1st d1 lap-time: ', fin_time - init_time)
        # pdb.set_trace()
    if model2 is None:
        results_filename = "results/results_visualize_sharpness_sparsity1_{}_d1_{}_v_{}_{}_{}_{}.csv".format(
            sparsity1, num_d, num_v, train_mode_str, parser_args.dataset, parser_args.algo)
    else:
        results_filename = "results/results_visualize_connectivity_d_{}_v_{}_resol_{}_{}_{}_{}_{}.csv".format(
            num_d, num_v, resol, train_mode_str, parser_args.dataset, parser_args.algo, parser_args.interpolate)

    results_df.to_csv(results_filename, index=False)


# connect two weights trained by "weight_training"
def connect_weight(model, criterion, data, validate, model2=None):
    # concatenate the weights
    flat_weight = []
    for name, params in model.named_parameters():
        if ".weight" in name:
            flat_weight.append(params.data)
    weight_init = _flatten_dense_tensors(flat_weight)

    flat_weight2 = []
    for name, params in model2.named_parameters():
        if ".weight" in name:
            flat_weight2.append(params.data)
    weight_fin = _flatten_dense_tensors(flat_weight2)

    num_d = 1  # 100
    num_v = 5  # 100
    resol = 100  # 1000

    if parser_args.interpolate == 'linear':
        num_v = 1

    # batch data to test
    for data_, label_ in data.train_loader:
        data_, label_ = data_.cuda(), label_.cuda()
        break

    # sanity check on the input model
    '''
    init_loss = criterion(model(data_), label_)
    print(init_loss.data.item())
    init_loss2 = criterion(model2(data_), label_)
    print(init_loss2.data.item())
    '''

    # setting for saving results
    cp_model = copy.deepcopy(model)
    dist_list = []
    train_mode_str = 'weight_training' if parser_args.weight_training else 'pruning'

    # init_time = time.time()
    for d1_idx in range(num_d):
        train_loss_mean_list = []
        train_loss_std_list = []
        train_acc_mean_list = []
        train_acc_std_list = []
        test_acc_mean_list = []
        test_acc_std_list = []
        if model2 is None:
            raise NotImplementedError
        else:
            weight_dest = weight_fin

        for i in range(resol+1):
            p = i/resol  # probability of sampling dest
            if d1_idx == 0:
                if model2 is None:
                    dist_list.append(round(p * sparsity1, 4))
                else:
                    dist_list.append(round(p, 4))

            # loss_avg = 0
            # acc_avg = 0
            loss_arr, train_acc_arr, acc_arr = np.zeros(
                num_v), np.zeros(num_v), np.zeros(num_v)

            for v_idx in range(num_v):
                if parser_args.interpolate == 'prob':
                    # [0, 1]^n  0 : I'll sample weight_init, 1: I'll sample weight_dest
                    sampling_vct = torch.bernoulli(
                        torch.ones_like(weight_init) * p)
                    new_weight = weight_init * \
                        (1-sampling_vct) + weight_dest * sampling_vct  # w+v
                elif parser_args.interpolate == 'linear':
                    new_weight = weight_init * (1-p) + weight_dest * p

                # put merged masks back to the model
                new_weight_unflat = _unflatten_dense_tensors(
                    new_weight, flat_weight)
                idx = 0
                for name, params in cp_model.named_parameters():
                    if ".weight" in name:
                        params.data = new_weight_unflat[idx]
                        idx += 1

                # compute loss for the mask
                loss = criterion(cp_model(data_), label_)
                acc1, acc5, acc10 = validate(
                    data.val_loader, cp_model, criterion, parser_args,
                    writer=None, epoch=parser_args.start_epoch)

                train_acc1, train_acc5, train_acc10 = validate(
                    data.train_loader, cp_model, criterion, parser_args,
                    writer=None, epoch=parser_args.start_epoch)

                print(i, v_idx, loss.data.item(), acc1, train_acc1)
                loss_arr[v_idx] = loss.data.item()
                acc_arr[v_idx] = acc1
                train_acc_arr[v_idx] = train_acc1

            train_loss_mean_list.append(np.mean(loss_arr))
            train_loss_std_list.append(np.std(loss_arr))
            train_acc_mean_list.append(np.mean(train_acc_arr))
            train_acc_std_list.append(np.std(train_acc_arr))
            test_acc_mean_list.append(np.mean(acc_arr))
            test_acc_std_list.append(np.std(acc_arr))

        if d1_idx == 0:
            results_df = pd.DataFrame({'dist': dist_list, 'train_loss_mean': train_loss_mean_list, 'train_loss_std': train_loss_std_list,
                                       'train_acc_mean': train_acc_mean_list, 'train_acc_std': train_acc_std_list,
                                       'test_acc_mean': test_acc_mean_list, 'test_acc_std': test_acc_std_list})
        else:
            raise NotImplementedError
            # results_df['batch_train_loss{}'.format(d1_idx+1)] = train_loss_list

        # fin_time = time.time()
        # print('1st d1 lap-time: ', fin_time - init_time)
    if model2 is None:
        results_filename = "results/results_visualize_sharpness_sparsity1_{}_d1_{}_v_{}_{}_{}_{}_{}.csv".format(
            sparsity1, num_d, num_v, train_mode_str, parser_args.dataset, parser_args.algo, parser_args.interpolate)
    else:
        results_filename = "results/results_visualize_connectivity_d_{}_v_{}_{}_{}_{}_{}.csv".format(
            num_d, num_v, train_mode_str, parser_args.dataset, parser_args.algo, parser_args.interpolate)

    results_df.to_csv(results_filename, index=False)


def visualize_mask_2D(model, criterion, data, validate):

    flat_tensor = []
    # concatenate the masks
    for name, params in model.named_parameters():
        if ".score" in name:
            flat_tensor.append(params.data)
    # a: flat_tensor, b = mask_init,
    mask_init = _flatten_dense_tensors(flat_tensor)

    # select random direction to go
    sparsity = 0.05
    num_d = 1  # 100
    num_v = 10  # 100
    resol = 1000  # 1000

    # batch data to test
    for data_, label_ in data.train_loader:
        data_, label_ = data_.cuda(), label_.cuda()
        break

    # setting for saving results
    cp_model = copy.deepcopy(model)
    train_mode_str = 'weight_training' if parser_args.weight_training else 'pruning'
    results_filename = "results/results_2D_visualize_sharpness_epoch_sparsity_{}_d_{}_v_{}_{}_{}_{}".format(
        sparsity, num_d, num_v, train_mode_str, parser_args.dataset, parser_args.algo)

    # init_time = time.time()
    for d1_idx in range(num_d):

        d1 = torch.bernoulli(torch.ones_like(mask_init) * sparsity)  # d1
        print('sum of d1: ', torch.sum(d1))
        d2 = torch.bernoulli(torch.ones_like(mask_init) * sparsity)  # d2
        print('sum of d2: ', torch.sum(d2))
        print('sum of d1*d2: ', torch.sum(d1*d2))

        # new_d1 = (d1 + mask_init) % 2
        # new_d2 = (d2 + mask_init) % 2

        loss_arr = np.zeros((resol, resol))
        for i1 in range(resol):
            p1 = i1/resol  # probability of adding elements from d1
            for i2 in range(resol):
                p2 = i2/resol  # probability of adding elements from d2
                loss_avg = 0

                for v_idx in range(num_v):
                    # [0, 1]^n  1: I'll add d1 elements
                    sampling_vct1 = torch.bernoulli(
                        torch.ones_like(mask_init) * p1)
                    # [0, 1]^n  1: I'll add d2 elements
                    sampling_vct2 = torch.bernoulli(
                        torch.ones_like(mask_init) * p2)

                    new_mask = (mask_init + sampling_vct1 * d1 +
                                sampling_vct2 * d2) % 2  # w+v1+v2

                    # put merged masks back to the model
                    new_mask_unflat = _unflatten_dense_tensors(
                        new_mask, flat_tensor)
                    idx = 0
                    for name, params in cp_model.named_parameters():
                        if ".score" in name:
                            params.data = new_mask_unflat[idx]
                            idx += 1

                    # compute loss for the mask
                    loss = criterion(cp_model(data_), label_)
                    # print(i1, i2, v_idx, loss.data.item())
                    loss_avg += loss.data.item()
                loss_arr[i1, i2] = loss_avg/num_v

        # print(loss_arr)
        np.save(results_filename + "_{}.npy".format(d1_idx), loss_arr)
        saved_loss = np.load(results_filename + "_{}.npy".format(d1_idx))
        print('saved_loss for d1_idx {}'.format(d1_idx), saved_loss)
    #     if d1_idx == 0:
    #         results_df = pd.DataFrame({'dist': dist_list, 'batch_train_loss': train_loss_list})
    #     else:
    #         results_df['batch_train_loss{}'.format(d1_idx+1)] = train_loss_list

    #     #fin_time = time.time()
    #     #print('1st d1 lap-time: ', fin_time - init_time)
    #     #pdb.set_trace()
    # results_df.to_csv(results_filename, index=False)


def get_trainer(parser_args):
    print(f"=> Using trainer from trainers.{parser_args.trainer}")
    trainer = importlib.import_module(f"trainers.{parser_args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(parser_args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if parser_args.gpu is not None:
        torch.cuda.set_device(parser_args.gpu)
        model.cuda(parser_args.gpu)

        if parser_args.multiprocessing_distributed:
            torch.distributed.init_process_group(
                backend=parser_args.dist_backend,
                init_method='env://',
                world_size=parser_args.world_size,
                rank=parser_args.rank
            )
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[parser_args.gpu], find_unused_parameters=True)
    else:
        device = torch.device("cpu")

    return model


def resume(parser_args, model, optimizer):
    if os.path.isfile(parser_args.resume):
        print(f"=> Loading checkpoint '{parser_args.resume}'")

        checkpoint = torch.load(
            parser_args.resume, map_location=f"cuda:{parser_args.gpu}")
        #if parser_args.start_epoch is None:
        #    print(f"=> Setting new start epoch at {checkpoint['epoch']}")
        #    parser_args.start_epoch = checkpoint["epoch"]

        #best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint)

        # optimizer.load_state_dict(checkpoint["optimizer"])

        #print(
        #    f"=> Loaded checkpoint '{parser_args.resume}' (epoch {checkpoint['epoch']})")

        return 0
    else:
        print(f"=> No checkpoint found at '{parser_args.resume}'")


def pretrained(path, model):
    if os.path.isfile(path):
        print("=> loading pretrained weights from '{}'".format(path))
        model.load_state_dict(torch.load(
            path, map_location=torch.device("cuda:{}".format(parser_args.gpu))))
        model.eval()
        '''
        pretrained = torch.load(path, map_location=torch.device("cuda:0"))["state_dict"]                 #map_location=torch.device("cuda:{}".format(parser_args.multigpu[0])),

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)
        '''

    else:
        print("=> no pretrained weights found at '{}'".format(path))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


def get_dataset(parser_args):
    print(f"=> Getting {parser_args.dataset} dataset")
    dataset = getattr(data, parser_args.dataset)(parser_args)

    return dataset


def get_model(parser_args):
    if parser_args.first_layer_dense:
        parser_args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(parser_args.arch))
    if parser_args.fixed_init:
        set_seed(parser_args.seed_fixed_init)
    if parser_args.arch in ['Conv4', 'Conv4Normal']:
        model = models.__dict__[parser_args.arch](width=parser_args.width)
    else:
        model = models.__dict__[parser_args.arch]()
    if parser_args.fixed_init:
        set_seed(parser_args.seed)

    if not parser_args.weight_training:
        # applying sparsity to the network
        if (
            parser_args.conv_type != "DenseConv"
            and parser_args.conv_type != "SampleSubnetConv"
            and parser_args.conv_type != "ContinuousSparseConv"
        ):
            if parser_args.prune_rate < 0:
                raise ValueError("Need to set a positive prune rate")

            set_model_prune_rate(model, prune_rate=parser_args.prune_rate)
            print(
                f"=> Rough estimate model params {sum(int(p.numel() * (1-parser_args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
            )

        # freezing the weights if we are only doing subnet training
        if parser_args.freeze_weights:
            freeze_model_weights(model)

    return model


def get_optimizer(optimizer_args, model, finetune_flag=False):
    '''
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)
    '''
    if finetune_flag:
        opt_algo = optimizer_args.fine_tune_optimizer
        opt_lr = optimizer_args.fine_tune_lr
        opt_wd = optimizer_args.fine_tune_wd
    else:
        opt_algo = optimizer_args.optimizer
        opt_lr = optimizer_args.lr
        opt_wd = optimizer_args.wd
    if opt_algo == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if (
            "bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if (
            "bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if optimizer_args.no_bn_decay else opt_wd,
                },
                {"params": rest_params, "weight_decay": opt_wd},
            ],
            opt_lr,
            momentum=optimizer_args.momentum,
            weight_decay=opt_wd,
            nesterov=optimizer_args.nesterov,
        )
    elif opt_algo == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr,
            weight_decay=opt_wd
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(parser_args):
    if parser_args.config is None or parser_args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(parser_args.config).stem
    if parser_args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{parser_args.name}/prune_rate={parser_args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{parser_args.log_dir}/{config}/{parser_args.name}/prune_rate={parser_args.prune_rate}"
        )
    if parser_args.width_mult != 1.0:
        run_base_dir = run_base_dir / \
            "width_mult={}".format(str(parser_args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(parser_args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Current Val Top 10, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Val Top 10, "
            "Best Train Top 1, "
            "Best Train Top 5"
            "Best Train Top 10\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{curr_acc10:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_acc10:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
                "{best_train_acc10:.02f}\n"
            ).format(now=now, **kwargs)
        )


def print_num_dataset(data):
    num_train = 0
    for idx, (img, label) in enumerate(data.train_loader):
        num_train += label.size()[0]

    num_val = 0
    for idx, (img, label) in enumerate(data.val_loader):
        num_val += label.size()[0]

    num_test = 0
    for idx, (img, label) in enumerate(data.test_loader):
        num_test += label.size()[0]

    print(num_train, num_val, num_test)
