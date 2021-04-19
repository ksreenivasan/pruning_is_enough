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

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    hc_round,
    get_model_sparsity
)
from utils.schedulers import get_policy
from utils.utils import set_seed, plot_histogram_scores

import importlib

import data
import models

import copy
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def main():
    print(parser_args)

    set_seed(parser_args.seed)

    # Simply call main_worker function
    main_worker()


def main_worker():
    parser_args.gpu = None
    train, validate, modifier = get_trainer(parser_args)
    if parser_args.gpu is not None:
        print("Use GPU: {} for training".format(parser_args.gpu))

    for i in range(1):
        # create model and optimizer
        model = get_model(parser_args)
        model = set_gpu(parser_args, model)

        if parser_args.pretrained:
            pretrained(parser_args.pretrained, model)
        if parser_args.pretrained2:
            model2 = copy.deepcopy(model)
            # model2.load_state_dict(torch.load(parser_args.pretrained2)['state_dict'])
            pretrained(parser_args.pretrained2, model2)

        optimizer = get_optimizer(parser_args, model)
        data = get_dataset(parser_args)
        lr_policy = get_policy(parser_args.lr_policy)(optimizer, parser_args)

        if parser_args.label_smoothing is None:
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            criterion = LabelSmoothing(smoothing=parser_args.label_smoothing)

        # optionally resume from a checkpoint
        best_acc1 = 0.0
        best_acc5 = 0.0
        best_acc10 = 0.0
        best_train_acc1 = 0.0
        best_train_acc5 = 0.0
        best_train_acc10 = 0.0

        if parser_args.resume:
            best_acc1 = resume(parser_args, model, optimizer)

        if parser_args.evaluate:
            acc1, acc5, acc10 = validate(
                data.val_loader, model, criterion, parser_args,
                writer=None, epoch=parser_args.start_epoch)
            print('Performance of model')
            print('acc1: {}, acc5: {}, acc10: {}'.format(acc1, acc5, acc10))

            acc1, acc5, acc10 = validate(
                data.val_loader, model2, criterion, parser_args,
                writer=None, epoch=parser_args.start_epoch)
            print('Performance of model2')
            print('acc1: {}, acc5: {}, acc10: {}'.format(acc1, acc5, acc10))

            for trial in range(parser_args.num_test):
                cp_model = copy.deepcopy(model)
                if parser_args.algo in ['hc']:
                    hc_round(cp_model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio)
                    # get_score_sparsity_hc(cp_model)

                    acc1, acc5, acc10 = validate(
                        data.val_loader, cp_model, criterion,
                        parser_args, writer=None, epoch=parser_args.start_epoch
                    )
                    print('Performance of model after pruning')
                    print('acc1: {}, acc5: {}, acc10: {}'.format(acc1, acc5, acc10))

            if parser_args.pretrained2:
                cp_model2 = copy.deepcopy(model2)
                if parser_args.algo in ['hc']:
                    hc_round(cp_model2, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio)
                    # get_score_sparsity_hc(cp_model2)

                    acc1, acc5, acc10 = validate(
                        data.val_loader, cp_model2, criterion,
                        parser_args, writer=None, epoch=parser_args.start_epoch
                    )
                    print('Performance of model2 after pruning')
                    print('acc1: {}, acc5: {}, acc10: {}'.format(acc1, acc5, acc10))
            else:
                cp_model2 = None

            if parser_args.weight_training:
                print('We are connecting weights')
                connect_weight(cp_model, criterion, data, validate, cp_model2)
            elif parser_args.algo in ['hc', 'ep']:
                print('We are connecting masks')
                connect_mask(cp_model, criterion, data, validate, cp_model2)
            # visualize_mask_2D(cp_model, criterion, data, validate)

            return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(parser_args)
    parser_args.ckpt_base_dir = ckpt_base_dir

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    parser_args.start_epoch = parser_args.start_epoch or 0
    acc1 = None

    epoch_list = []
    test_acc_list = []
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

    # sanity check for 50% sparsity initialization
    ## if parser_args.score_init == 'bern':
       ## get_score_sparsity_hc(model)

    # Start training
    for epoch in range(parser_args.start_epoch, parser_args.epochs):
        lr_policy(epoch, iteration=None)
        modifier(parser_args, epoch, model)

        cur_lr = get_lr(optimizer)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5, train_acc10 = train(
            data.train_loader, model, criterion, optimizer, epoch, parser_args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)



        # apply round for every T epochs (after E warm-up epoch)
        if epoch >= parser_args.hc_warmup and epoch % parser_args.hc_period == 0:
            print('Apply rounding: {}'.format(parser_args.round))
            hc_round(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio)

        # evaluate on validation set
        start_validation = time.time()
        if parser_args.algo in ['hc']:
            cp_model = copy.deepcopy(model)
            hc_round(cp_model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio)
            acc1, acc5, acc10 = validate(data.val_loader, cp_model, criterion, parser_args, writer, epoch)
        else:
            acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # save the histrogram of scores
        if not parser_args.weight_training:
            if epoch % 5 == 1:  # %10 %50
                plot_histogram_scores(model, parser_args.algo, epoch)
                print('Plotted the score histogram')

        if not parser_args.weight_training:
            avg_sparsity = get_model_sparsity(model)
        else:
            # haven't written a weight sparsity function yet
            avg_sparsity = -1
        # update all results lists
        epoch_list.append(epoch)
        test_acc_list.append(acc1)
        # TODO: define sparsity for cifar10 networks
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

    # TODO: plot histograms here too
    results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc': test_acc_list, 'model_sparsity': model_sparsity_list})
    if parser_args.results_filename:
        results_filename = parser_args.results_filename
    else:
        # TODO: move this to utils
        train_mode_str = 'weight_training' if parser_args.weight_training else 'pruning'
        results_filename = "results/results_acc_{}_{}_{}.csv".format(train_mode_str, parser_args.dataset, parser_args.algo)
    results_df.to_csv(results_filename, index=False)

    # sanity check whether the weight values did not change
    for name, params in model.named_parameters():
        if ".weight" in name:
            print(torch.sum(params.data))

    # check the performance of trained model
    if parser_args.algo in ['hc']:
        for trial in range(parser_args.num_round):
            cp_model = copy.deepcopy(model)
            print('Apply rounding for the final model:')
            hc_round(cp_model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio)
            # hc_round(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio)
            # get_score_sparsity_hc(cp_model)

            acc1, acc5, acc10 = validate(
                data.val_loader, cp_model, criterion,
                parser_args, writer=None, epoch=parser_args.start_epoch
            )
            print('acc1: {}, acc5: {}, acc10: {}'.format(acc1, acc5, acc10))


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
    num_v = 5  # 5 # 100
    resol = 100  # 1000

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
            sparsity1 = 0.2
            d1 = torch.bernoulli(torch.ones_like(mask_init) * sparsity1) # d1
            # print('sum of d1: ', torch.sum(d1))
            new_d1 = (d1 + mask_init) % 2
        else:
            new_d1 = mask_fin
        normalized_hamming_dist = (torch.sum(torch.abs(mask_init - new_d1))/len(mask_init)).data.item()
        print('dist btw mask_init and new_d1: ', normalized_hamming_dist)

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
                sampling_vct = torch.bernoulli(torch.ones_like(mask_init) * p)  # [0, 1]^n  0 : I'll sample mask_init, 1: I'll sample d1
                new_mask = mask_init * (1-sampling_vct) + new_d1 * sampling_vct   # w+v

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
        results_filename = "results/results_visualize_connectivity_d_{}_v_{}_{}_{}_{}_{}.csv".format(num_d, num_v, train_mode_str, parser_args.dataset, parser_args.algo, parser_args.interpolate)

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
        model = model.cuda(parser_args.gpu)
    elif parser_args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {parser_args.multigpu} gpus")
        torch.cuda.set_device(parser_args.multigpu[0])
        parser_args.gpu = parser_args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=parser_args.multigpu).cuda(
            parser_args.multigpu[0]
        )

    cudnn.benchmark = True

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
        pretrained = torch.load(
            path,
            map_location=torch.device("cuda:0"), #map_location=torch.device("cuda:{}".format(parser_args.multigpu[0])),
        )["state_dict"]

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
        set_seed(parser_args.seed)
    model = models.__dict__[parser_args.arch]() #model = models.__dict__[parser_args.arch](shift=parser_args.shift)
    if parser_args.fixed_init:    
        set_seed(parser_args.seed2)
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


def get_optimizer(parser_args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if parser_args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if parser_args.no_bn_decay else parser_args.weight_decay,
                },
                {"params": rest_params, "weight_decay": parser_args.weight_decay},
            ],
            parser_args.lr,
            momentum=parser_args.momentum,
            weight_decay=parser_args.weight_decay,
            nesterov=parser_args.nesterov,
        )
    elif parser_args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=parser_args.lr,
            weight_decay=parser_args.weight_decay
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
