from args import args as parser_args
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
)
from utils.schedulers import get_policy
from utils.utils import set_seed, plot_histogram_scores

import importlib

import data
import models

import copy
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def eval_and_print(validate, data_loader, model, criterion, parser_args, writer=None, epoch=parser_args.start_epoch, description='model'):

    acc1, acc5, acc10 = validate(data_loader, model, criterion, parser_args, writer=None, epoch=parser_args.start_epoch)
    print('Performance of {}'.format(description))
    print('acc1: {}, acc5: {}, acc10: {}'.format(acc1, acc5, acc10))

    return acc1

def finetune(model, parser_args, data, criterion, old_epoch_list, old_test_acc_before_round_list, old_test_acc_list, old_reg_loss_list, old_model_sparsity_list, result_root, shuffle=False, reinit=False, invert=False, chg_mask=False, chg_weight=False):
    epoch_list = copy.deepcopy(old_epoch_list)
    test_acc_before_round_list = copy.deepcopy(old_test_acc_before_round_list)
    test_acc_list = copy.deepcopy(old_test_acc_list)
    reg_loss_list = copy.deepcopy(old_reg_loss_list)
    model_sparsity_list = copy.deepcopy(old_model_sparsity_list)

    '''
    # round the score (in the model itself)
    model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)    
    # apply reinit/shuffling masks/weights (if necessary)
    model = redraw(model, shuffle=shuffle, reinit=reinit, invert=invert, chg_mask=chg_mask, chg_weight=chg_weight)

    # switch to weight training mode (turn on the requires_grad for weight/bias, and turn off the requires_grad for other parameters)
    model = switch_to_wt(model)
    '''    

    # set base_setting and evaluate 
    run_base_dir, ckpt_base_dir, log_base_dir, writer, epoch_time, validation_time, train_time, progress_overall = get_settings(parser_args)
    optimizer = get_optimizer(parser_args, model, finetune_flag=True)
    train, validate, modifier = get_trainer(parser_args)

    # print the accuracy right after beginning finetune function
    acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer, parser_args.epochs-1)
    print('accuracy right after beginning finetune function: ', acc1)


    if parser_args.algo in ['hc', 'hc_iter']:
        # round the score (in the model itself)
        model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)    
        acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer, parser_args.epochs-1)
        print('accuracy after rounding scores: ', acc1)

        # apply reinit/shuffling masks/weights (if necessary)
        model = redraw(model, shuffle=shuffle, reinit=reinit, invert=invert, chg_mask=chg_mask, chg_weight=chg_weight)
        acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer, parser_args.epochs-1)
        print('accuracy after redrawing masks/weights: ', acc1)

    # switch to weight training mode (turn on the requires_grad for weight/bias, and turn off the requires_grad for other parameters)
    model = switch_to_wt(model)

    # check the performance of loaded model (after rounding)
    acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer, parser_args.epochs-1)
    avg_sparsity = model_sparsity_list[-1] # copy & paste the sparsity of prev. epoch
    epoch_list.append(parser_args.epochs-1)
    test_acc_before_round_list.append(-1)
    test_acc_list.append(acc1)
    reg_loss_list.append(0.0)
    model_sparsity_list.append(avg_sparsity)

    print('accuracy after switching to weight training: ', acc1)
    #pdb.set_trace()


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
        acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)
        avg_sparsity = model_sparsity_list[-1] # copy & paste the sparsity of prev. epoch

        # update all results lists
        epoch_list.append(epoch)
        test_acc_before_round_list.append(-1)
        test_acc_list.append(acc1)
        reg_loss_list.append(reg_loss)
        model_sparsity_list.append(avg_sparsity)


        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )
        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

        results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc_before_rounding': test_acc_before_round_list,'test_acc': test_acc_list,
                                   'regularization_loss': reg_loss_list, 'model_sparsity': model_sparsity_list})
        if not chg_mask and not chg_weight:
            results_filename = result_root + 'acc_and_sparsity.csv'    
        #elif chg_weight and shuffle:
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
    fan_str = parser_args.scale_fan
    w_str = parser_args.init
    s_str = parser_args.score_init
    width_str = parser_args.width
    seed_str = parser_args.seed + parser_args.trial_num - 1
    idty_str = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_fan_{}_{}_{}_width_{}_seed_{}".\
        format(train_mode_str, dataset_str, model_str, algo_str, rate_str, period_str, reg_str, reg_lmbda,
        opt_str, policy_str, lr_str, lr_gamma, lr_adj, fan_str, w_str, s_str,
        width_str, seed_str).replace(".", "_")


    return idty_str

def get_settings(parser_args):

    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(parser_args)
    parser_args.ckpt_base_dir = ckpt_base_dir
    writer = SummaryWriter(log_dir=log_base_dir)
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
        acc1 = eval_and_print(validate, data_loader, prob_model, criterion, parser_args, description='probabilistic model {}'.format(i))
        idx_list.append(i)
        test_acc_list.append(acc1)

        prob_mask, _ = get_mask(prob_model)
        hamm_dist = torch.sum(torch.abs(naive_mask - prob_mask))/len(naive_mask)
        dist_list.append(hamm_dist.data.item())
        mask_list.append(prob_mask.data)

    # save the result in the dataframe
    compare_df = pd.DataFrame({'idx': idx_list, 'test_acc': test_acc_list, 'hamming dist to naive': dist_list})
    results_filename = result_root + 'compare_rounding.csv'
    print("Writing rounding compare results into: {}".format(results_filename))
    compare_df.to_csv(results_filename, index=False)

    compare_prob = np.zeros((10, 10))
    for i in range(n_rand):
        for j in range(n_rand):
            compare_prob[i, j] = torch.sum(torch.abs(mask_list[i] - mask_list[j])/len(mask_list[i]))
    print(compare_prob)
    pd.DataFrame(compare_prob).to_csv(result_root + 'compare_probs.csv', header=None, index=False)


    return


# switches off gradients for scores and flags and switches it on for weights and biases
def switch_to_wt(model):
    print('Switching to weight training by switching off requires_grad for scores and switching it on for weights.')

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
    mask = _flatten_dense_tensors(flat_tensor)  # a: flat_tensor, b = mask_init,

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

def main():
    print(parser_args)
    set_seed(parser_args.seed + parser_args.trial_num - 1)

    # parser_args.distributed = parser_args.world_size > 1 or parser_args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    
    if parser_args.multiprocessing_distributed:
        setup_distributed(ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,), join=True)
    else:
        # Simply call main_worker function
        main_worker(parser_args.gpu, ngpus_per_node)


def main_worker(gpu, ngpus_per_node):
    train, validate, modifier = get_trainer(parser_args)
    parser_args.gpu = gpu

    if parser_args.gpu is not None:
        print("Use GPU: {} for training".format(parser_args.gpu))

    if parser_args.multiprocessing_distributed:
        parser_args.rank = parser_args.rank * ngpus_per_node + parser_args.gpu
        # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        parser_args.batch_size = int(parser_args.batch_size / ngpus_per_node)
        parser_args.num_workers = int((parser_args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        parser_args.world_size = ngpus_per_node * parser_args.world_size
    idty_str = get_idty_str(parser_args)
    result_root = 'results/results_' + idty_str + '/'
    if not os.path.isdir(result_root):
        os.mkdir(result_root)

    # create model and optimizer
    model = get_model(parser_args)
    if parser_args.weight_training:
        model = switch_to_wt(model) 
    model = set_gpu(parser_args, model)
    if parser_args.pretrained:
        pretrained(parser_args.pretrained, model)
    if parser_args.pretrained2:
        model2 = copy.deepcopy(model)  # model2.load_state_dict(torch.load(parser_args.pretrained2)['state_dict'])
        pretrained(parser_args.pretrained2, model2)
    optimizer = get_optimizer(parser_args, model)
    data = get_dataset(parser_args)
    lr_policy = get_policy(parser_args.lr_policy)(optimizer, parser_args)

    if parser_args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=parser_args.label_smoothing)
        # if isinstance(model, nn.parallel.DistributedDataParallel):
        #     model = model.module
    

    if parser_args.random_subnet:
        # round the score (in the model itself)
        model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)    
        
        # TODO: CHANGE THIS BACK once the finetune from checkpoints code is fixed
        # NOTE: this part is hard coded
        model = redraw(model, shuffle=parser_args.shuffle, reinit=parser_args.reinit, chg_mask=parser_args.chg_mask, chg_weight=parser_args.chg_weight)  

        # switch to weight training mode (turn on the requires_grad for weight/bias, and turn off the requires_grad for other parameters)
        model = switch_to_wt(model)

        # set base_setting and evaluate 
        run_base_dir, ckpt_base_dir, log_base_dir, writer, epoch_time, validation_time, train_time, progress_overall = get_settings(parser_args)
        # TODO: Change this to use finetune() (I think this is possible)
        optimizer = get_optimizer(parser_args, model, finetune_flag=True)
        train, validate, modifier = get_trainer(parser_args)

        # check the performance of loaded model (after rounding)
        acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer, parser_args.epochs-1)
        epoch_list = []
        test_acc_before_round_list = []
        test_acc_list = []
        reg_loss_list = []
        model_sparsity_list = []

        for epoch in range(parser_args.epochs):

            if parser_args.multiprocessing_distributed:
                data.train_loader.sampler.set_epoch(epoch)
            #lr_policy(epoch, iteration=None)
            #modifier(parser_args, epoch, model)
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
            acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)
            cp_model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)
            avg_sparsity = get_model_sparsity(cp_model)
            print('Model avg sparsity: {}'.format(avg_sparsity))

            # update all results lists
            epoch_list.append(epoch)
            test_acc_before_round_list.append(-1)
            test_acc_list.append(acc1)
            reg_loss_list.append(reg_loss)
            model_sparsity_list.append(avg_sparsity)

            epoch_time.update((time.time()) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )
            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()

            results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc_before_rounding': test_acc_before_round_list,'test_acc': test_acc_list, 'regularization_loss': reg_loss_list, 'model_sparsity': model_sparsity_list})

            if parser_args.results_filename:
                results_filename = parser_args.results_filename
            else:
                results_filename = result_root + 'random_subnet_{}.csv'.format(parser_args.prune_rate)
            print("Writing results into: {}".format(results_filename))
            results_df.to_csv(results_filename, index=False)

        if parser_args.multiprocessing_distributed:
            cleanup_distributed()

        # save checkpoint for later debug
        model_filename = "random_subnet_finetuned_{}_ckpt.pt".format(parser_args.prune_rate)
        print("Writing final model to {}".format(model_filename))
        torch.save(model.state_dict(), model_filename)

        return


    best_acc1, best_acc5, best_acc10, best_train_acc1, best_train_acc5, best_train_acc10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # optionally resume from a checkpoint
    if parser_args.resume:
        best_acc1 = resume(parser_args, model, optimizer)
    # when we only evaluate a pretrained model
    if parser_args.evaluate:
        #eval_and_print(validate, data.val_loader, model, criterion, parser_args, writer=None, epoch=parser_args.start_epoch, description='model')
        if parser_args.algo in ['hc_iter']:

            model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)
            eval_and_print(validate, data.val_loader, model, criterion, parser_args, writer=None, epoch=parser_args.start_epoch, description='final model after rounding')
            # sanity_check(model, parser_args, data, criterion)

        for trial in range(parser_args.num_test):
            if parser_args.algo in ['hc']:
                if parser_args.how_to_connect == "prob":
                    cp_model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)
                else:
                    cp_model = copy.deepcopy(model)
                eval_and_print(validate, data.val_loader, cp_model, criterion, parser_args, writer=None, epoch=parser_args.start_epoch, description='model after pruning')

        if parser_args.pretrained2:
            eval_and_print(validate, data.val_loader, model2, criterion, parser_args, writer=None, epoch=parser_args.start_epoch, description='model2')
            if parser_args.algo in ['hc']:
                if parser_args.how_to_connect == "prob":
                    cp_model2 = round_model(model2, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)
                else:
                    cp_model2 = copy.deepcopy(model2)
                eval_and_print(validate, data.val_loader, cp_model2, criterion, parser_args, writer=None, epoch=parser_args.start_epoch, description='model2 after pruning')

        if parser_args.pretrained and parser_args.pretrained2 and parser_args.mode_connect:
            if parser_args.weight_training:
                print('We are connecting weights')
                connect_weight(cp_model, criterion, data, validate, cp_model2)
            elif parser_args.algo in ['hc', 'ep']:
                print('We are connecting masks')
                connect_mask(cp_model, criterion, data, validate, cp_model2)
            # visualize_mask_2D(cp_model, criterion, data, validate)

        return

    # Set up directories & setting
    run_base_dir, ckpt_base_dir, log_base_dir, writer, epoch_time, validation_time, train_time, progress_overall = get_settings(parser_args)
    end_epoch = time.time()
    parser_args.start_epoch = parser_args.start_epoch or 0
    acc1 = None
    epoch_list = []
    test_acc_before_round_list = []
    test_acc_list = []
    reg_loss_list = []
    model_sparsity_list = []

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": parser_args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_acc10": best_acc10,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "best_train_acc10": best_train_acc10,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    # Start training
    for epoch in range(parser_args.start_epoch, parser_args.epochs):
        if parser_args.multiprocessing_distributed:
            data.train_loader.sampler.set_epoch(epoch)
        lr_policy(epoch, iteration=None)
        modifier(parser_args, epoch, model)
        cur_lr = get_lr(optimizer)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5, train_acc10, reg_loss = train(
            data.train_loader, model, criterion, optimizer, epoch, parser_args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # apply round for every T_{round} epochs (after E warm-up epoch)
        # if parser_args.algo in ['hc', 'hc_iter'] and epoch >= parser_args.hc_warmup and epoch % parser_args.hc_period == 0:
        #     print('Apply rounding: {}'.format(parser_args.round))
        #     model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)

        # evaluate on validation set
        start_validation = time.time()
        if parser_args.algo in ['hc', 'hc_iter']:
            br_acc1, br_acc5, br_acc10 = validate(data.val_loader, model, criterion, parser_args, writer, epoch) # before rounding
            print('Acc before rounding: {}'.format(br_acc1))
            acc_avg = 0
            for num_trial in range(parser_args.num_test):
                cp_model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)
                acc1, acc5, acc10 = validate(data.val_loader, cp_model, criterion, parser_args, writer, epoch)
                acc_avg += acc1
            acc_avg /= parser_args.num_test
            acc1 = acc_avg
            print('Acc after rounding: {}'.format(acc1))
        else:
            acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # save the histrogram of scores
        '''
        if not parser_args.weight_training:
            if (epoch % 25 == 1) or epoch == (parser_args.epochs-1):
                plot_histogram_scores(model, result_root+'Epoch_{}.pdf'.format(epoch), parser_args.arch)
        '''

        # prune the model every T_{prune} epochs
        if parser_args.algo == 'hc_iter' and epoch % (parser_args.iter_period) == 0 and epoch != 0:
            prune(model)
            if parser_args.checkpoint_at_prune:
                # let's see if we can get all sparsity plots with one run
                # save checkpoints at every pruned model so that we can finetune later
                # save checkpoint for later debug
                cp_model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)
                avg_sparsity = get_model_sparsity(cp_model)
                idty_str = get_idty_str(parser_args)
                ckpt_root = 'model_checkpoints/ckpts_' + idty_str + '/'
                if not os.path.isdir(ckpt_root):
                    os.mkdir(ckpt_root)
                model_filename = ckpt_root + "hc_ckpt_at_sparsity_{}.pt".format(int(avg_sparsity))
                print("Checkpointing model to {}".format(model_filename))
                torch.save(model.state_dict(), model_filename)

        # get model sparsity
        if not parser_args.weight_training:
            if parser_args.algo in ['hc', 'hc_iter']:
                # Round before checking sparsity
                cp_model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)
                avg_sparsity = get_model_sparsity(cp_model)
                print('Model avg sparsity: {}'.format(avg_sparsity))
            else:
                avg_sparsity = get_model_sparsity(model)
        else:
            # haven't written a weight sparsity function yet
            avg_sparsity = 1

        # update all results lists
        epoch_list.append(epoch)
        if parser_args.algo in ['hc', 'hc_iter']:
            test_acc_before_round_list.append(br_acc1)
        else:
            # no before rounding for EP/weight training
            test_acc_before_round_list.append(-1)
        test_acc_list.append(acc1)
        reg_loss_list.append(reg_loss)
        model_sparsity_list.append(avg_sparsity)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_acc10 = max(acc10, best_acc10)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        best_train_acc10 = max(train_acc10, best_train_acc10)

        save = ((epoch % parser_args.save_every) == 0) and parser_args.save_every > 0
        if is_best or save or epoch == parser_args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": parser_args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_acc10": best_acc10,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "best_train_acc10": best_train_acc10,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                    "curr_acc10": acc10,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
                parser_args=parser_args,
            )


        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        if parser_args.conv_type == "SampleSubnetConv":
            count = 0
            sum_pr = 0.0
            for n, m in model.named_modules():
                if isinstance(m, SampleSubnetConv):
                    # avg pr across 10 samples
                    pr = 0.0
                    for _ in range(10):
                        pr += (
                            (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
                            .float()
                            .mean()
                            .item()
                        )
                    pr /= 10.0
                    writer.add_scalar("pr/{}".format(n), pr, epoch)
                    sum_pr += pr
                    count += 1

            parser_args.prune_rate = sum_pr / count
            writer.add_scalar("pr/average", parser_args.prune_rate, epoch)

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

        if parser_args.algo in ['hc', 'hc_iter']:
            results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc_before_rounding': test_acc_before_round_list,'test_acc': test_acc_list, 'regularization_loss': reg_loss_list, 'model_sparsity': model_sparsity_list})
        else:
            results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc': test_acc_list, 'model_sparsity': model_sparsity_list})

        if parser_args.results_filename:
            results_filename = parser_args.results_filename
        else:
            results_filename = result_root + 'acc_and_sparsity.csv'    
        print("Writing results into: {}".format(results_filename))
        results_df.to_csv(results_filename, index=False)

    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_acc10=best_acc10,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        best_train_acc10=best_train_acc10,
        prune_rate=parser_args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        curr_acc10=acc10,
        base_config=parser_args.config,
        name=parser_args.name,
    )

    # check the performance of trained model
    if parser_args.algo in ['hc', 'hc_iter', 'ep']:
        cp_model = copy.deepcopy(model)
        if not parser_args.skip_fine_tune:
            print("Beginning fine-tuning")
            cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list, reg_loss_list, model_sparsity_list, result_root)
            # print out the final acc
            eval_and_print(validate, data.val_loader, cp_model, criterion, parser_args, writer=None, description='final model after finetuning')
        else:
            print("Skipping finetuning!!!")

        if not parser_args.skip_sanity_checks:
            print("Beginning Sanity Checks:")
            # do the sanity check for shuffled mask/weights, reinit weights
            print("Sanity Check 1: Weight Reinit")
            cp_model = copy.deepcopy(model)
            cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list,
                                reg_loss_list, model_sparsity_list, result_root, reinit=True, chg_weight=True)

            '''
            print("Sanity Check 2: Weight Reshuffle")
            cp_model = copy.deepcopy(model)
            cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list,
                                reg_loss_list, model_sparsity_list, result_root, shuffle=True, chg_weight=True)
            '''
            print("Sanity Check 2: Mask Reshuffle")
            cp_model = copy.deepcopy(model)
            cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list,
                                reg_loss_list, model_sparsity_list, result_root, shuffle=True, chg_mask=True)
            
            print("Sanity Check 3: Mask Invert")
            cp_model = copy.deepcopy(model)
            cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list, 
                                reg_loss_list, model_sparsity_list, result_root, invert=True, chg_mask=True)
        
        else:
            print("Skipping sanity checks!!!")

    if parser_args.multiprocessing_distributed:
        cleanup_distributed()


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
    mask_init = _flatten_dense_tensors(flat_tensor)  # a: flat_tensor, b = mask_init,

    flat_tensor2 = []
    idx = 0
    for name, params in model2.named_parameters():
        if ".weight" in name:
            print(name, torch.sum(torch.abs(flat_weight[idx] - params.data)))
            idx += 1
        if ".score" in name:
            flat_tensor2.append(params.data)
            print(name, params.data.shape)
    mask_fin = _flatten_dense_tensors(flat_tensor2)  # a: flat_tensor2, b = mask_fin,

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
        if model2 is None: # when 2nd model is not specified (use random direction)
            sparsity1 = 0.2
            d1 = torch.bernoulli(torch.ones_like(mask_init) * sparsity1) # d1
            # print('sum of d1: ', torch.sum(d1))
            new_d1 = (d1 + mask_init) % 2
        else:
            new_d1 = mask_fin
        normalized_hamming_dist = (torch.sum(torch.abs(mask_init - new_d1))/len(mask_init)).data.item()
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
            loss_arr, train_acc_arr, acc_arr = np.zeros(num_v), np.zeros(num_v), np.zeros(num_v)

            for v_idx in range(num_v):
                if parser_args.how_to_connect == "prob":
                    sampling_vct = torch.bernoulli(torch.ones_like(mask_init) * p)  # [0, 1]^n  0 : I'll sample mask_init, 1: I'll sample d1
                    new_mask = mask_init * (1-sampling_vct) + new_d1 * sampling_vct   # w+v
                else:
                    new_mask = mask_init * p + mask_fin * (1-p) 

                # pdb.set_trace()
                # print(torch.sum(torch.abs(new_mask - new_d1)))
                # put merged masks back to the model
                new_mask_unflat = _unflatten_dense_tensors(new_mask, flat_tensor)
                idx = 0
                for name, params in cp_model.named_parameters():
                    if ".score" in name:
                        params.data = new_mask_unflat[idx]
                        # print(name, params.data.shape)
                        # print(torch.sum(torch.abs(params.data - flat_tensor2[idx])))
                        idx += 1

                if parser_args.how_to_connect == "round":
                    cp_model = round_model(cp_model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)

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
        results_filename = "results/results_visualize_sharpness_sparsity1_{}_d1_{}_v_{}_{}_{}_{}.csv".format(sparsity1, num_d, num_v, train_mode_str, parser_args.dataset, parser_args.algo)
    else:
        results_filename = "results/results_visualize_connectivity_d_{}_v_{}_resol_{}_{}_{}_{}_{}.csv".format(num_d, num_v, resol, train_mode_str, parser_args.dataset, parser_args.algo, parser_args.interpolate)

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
            loss_arr, train_acc_arr, acc_arr = np.zeros(num_v), np.zeros(num_v), np.zeros(num_v)

            for v_idx in range(num_v):
                if parser_args.interpolate == 'prob':
                    sampling_vct = torch.bernoulli(torch.ones_like(weight_init) * p)  # [0, 1]^n  0 : I'll sample weight_init, 1: I'll sample weight_dest
                    new_weight = weight_init * (1-sampling_vct) + weight_dest * sampling_vct  # w+v
                elif parser_args.interpolate == 'linear':
                    new_weight = weight_init * (1-p) + weight_dest * p

                # put merged masks back to the model
                new_weight_unflat = _unflatten_dense_tensors(new_weight, flat_weight)
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
        results_filename = "results/results_visualize_sharpness_sparsity1_{}_d1_{}_v_{}_{}_{}_{}_{}.csv".format(sparsity1, num_d, num_v, train_mode_str, parser_args.dataset, parser_args.algo, parser_args.interpolate)
    else:
        results_filename = "results/results_visualize_connectivity_d_{}_v_{}_{}_{}_{}_{}.csv".format(num_d, num_v, train_mode_str, parser_args.dataset, parser_args.algo, parser_args.interpolate)

    results_df.to_csv(results_filename, index=False)


def visualize_mask_2D(model, criterion, data, validate):

    flat_tensor = []
    # concatenate the masks
    for name, params in model.named_parameters():
        if ".score" in name:
            flat_tensor.append(params.data)
    mask_init = _flatten_dense_tensors(flat_tensor)  # a: flat_tensor, b = mask_init,

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
    results_filename = "results/results_2D_visualize_sharpness_epoch_sparsity_{}_d_{}_v_{}_{}_{}_{}".format(sparsity, num_d, num_v, train_mode_str, parser_args.dataset, parser_args.algo)

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
                    sampling_vct1 = torch.bernoulli(torch.ones_like(mask_init) * p1)  # [0, 1]^n  1: I'll add d1 elements
                    sampling_vct2 = torch.bernoulli(torch.ones_like(mask_init) * p2)  # [0, 1]^n  1: I'll add d2 elements

                    new_mask = (mask_init + sampling_vct1 * d1 + sampling_vct2 * d2) % 2  # w+v1+v2

                    # put merged masks back to the model
                    new_mask_unflat = _unflatten_dense_tensors(new_mask, flat_tensor)
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
            model = nn.parallel.DistributedDataParallel(model, device_ids=[parser_args.gpu], find_unused_parameters=True)
    else:
        device = torch.device("cpu")

    return model


def resume(parser_args, model, optimizer):
    if os.path.isfile(parser_args.resume):
        print(f"=> Loading checkpoint '{parser_args.resume}'")

        checkpoint = torch.load(parser_args.resume, map_location=f"cuda:{parser_args.multigpu[0]}")
        if parser_args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            parser_args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{parser_args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{parser_args.resume}'")


def pretrained(path, model):
    if os.path.isfile(path):
        print("=> loading pretrained weights from '{}'".format(path))
        model.load_state_dict(torch.load(path, map_location=torch.device("cuda:{}".format(parser_args.gpu))))
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
    for name, params in model.named_parameters():
        if ".weight" in name:
            print(torch.sum(params.data))

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
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

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
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
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
        run_base_dir = run_base_dir / "width_mult={}".format(str(parser_args.width_mult))

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


if __name__ == "__main__":
    main()

