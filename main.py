# from args import args as parser_args
# import pdb
# import numpy as np
# import os
# import pathlib
# import random
# import time
# import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
# import torch.multiprocessing as mp

# import sys
# import re

# from utils.conv_type import FixedSubnetConv, SampleSubnetConv
# from utils.logging import AverageMeter, ProgressMeter
# from utils.net_utils import (
#     set_model_prune_rate,
#     freeze_model_weights,
#     save_checkpoint,
#     get_lr,
#     LabelSmoothing,
#     round_model,
#     get_model_sparsity,
#     prune,
#     redraw,
# )
# from utils.schedulers import get_policy
# from utils.utils import set_seed, plot_histogram_scores

# import importlib

# import data
# import models

# import copy
# from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from main_utils import *


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
    model = get_model(parser_args)
    if parser_args.weight_training:
        model = switch_to_wt(model) 
    model = set_gpu(parser_args, model)
    if parser_args.pretrained:
        pretrained(parser_args.pretrained, model)
    if parser_args.pretrained2:
        model2 = copy.deepcopy(model)  # model2.load_state_dict(torch.load(parser_args.pretrained2)['state_dict'])
        pretrained(parser_args.pretrained2, model2)
    else:
        model2 = None
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
        test_random_subnet(model, data, criterion, parser_args, writer, result_root)
        return


    best_acc1, best_acc5, best_acc10, best_train_acc1, best_train_acc5, best_train_acc10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # optionally resume from a checkpoint
    if parser_args.resume:
        best_acc1 = resume(parser_args, model, optimizer)
    # when we only evaluate a pretrained model
    if parser_args.evaluate:
        evaluate_without_training(parser_args, model, model2, validate, data, criterion)
        return

    # Set up directories & setting
    run_base_dir, ckpt_base_dir, log_base_dir, writer, epoch_time, validation_time, train_time, progress_overall = get_settings(parser_args)
    end_epoch = time.time()
    parser_args.start_epoch = parser_args.start_epoch or 0
    acc1 = None
    epoch_list, test_acc_before_round_list, test_acc_list, reg_loss_list, model_sparsity_list = [], [], [], [], []

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
        print('epoch: {}, lr: {}'.format(epoch, cur_lr))
        print("="*60)
        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5, train_acc10, reg_loss = train(
            data.train_loader, model, criterion, optimizer, epoch, parser_args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

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

        # prune the model every T_{prune} epochs
        if parser_args.algo == 'hc_iter' and epoch % (parser_args.iter_period) == 0 and epoch != 0:
            prune(model)
            if parser_args.checkpoint_at_prune:
                save_checkpoint_at_prune(model, parser_args)

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
            do_sanity_checks(model, parser_args, data, criterion, epoch_list, test_acc_before_round_list, test_acc_list, reg_loss_list, model_sparsity_list, result_root)            
        
        else:
            print("Skipping sanity checks!!!")

    if parser_args.multiprocessing_distributed:
        cleanup_distributed()




if __name__ == "__main__":
    main()

