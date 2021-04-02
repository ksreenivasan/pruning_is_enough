from __future__ import print_function
import os
import math
import random
import sys
import time
import pathlib
import shutil
import csv

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import torch.autograd as autograd

from utils import data
from utils.train_test import prune_weights, prune_activations, inference, train, simulated_annealing
from utils.net_utils import get_sparsity, flip, zero_one_loss
import models.greedy as models

from args import args

def main():
    print(args, "\n")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    device = get_device(args)
    data = get_dataset(args)
    model = get_model(args, data, device)
   
    if args.loss == "cross-entropy-loss":
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.loss == "zero-one-loss":
        criterion = zero_one_loss()
    else:
        raise NotImplementedError("Unsupported loss type ...")
   
    config = pathlib.Path(args.config).stem
    base_dir = pathlib.Path(f"./results/greedy/{args.name}/{config}")
    if not base_dir.exists():
        os.makedirs(base_dir)
    
    ckpt_dir = base_dir / "checkpoints"
    if not ckpt_dir.exists():
        os.makedirs(ckpt_dir)

    (base_dir / "args.txt").write_text(str(args))

    curr_and_best = {
        'curr_train_acc1': 0,
        'curr_train_acc5': 0,
        'best_train_acc1': 0,
        'best_train_acc5': 0,
        'curr_val_acc1': 0,
        'curr_val_acc5': 0,
        'best_val_acc1': 0,
        'best_val_acc5': 0
    }

    ckpt_path = pathlib.Path(args.load_ckpt) if args.load_ckpt is not None else None

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        pretrain_state_dict = ckpt['state_dict']
        model.load_state_dict(pretrain_state_dict)

        start_epoch = ckpt['epoch'] + 1
        
        curr_and_best['best_train_acc1'] = ckpt['best_train_acc1']
        curr_and_best['best_train_acc5'] = ckpt['best_train_acc5']
        curr_and_best['best_val_acc1'] = ckpt['best_acc1']
        curr_and_best['best_val_acc5'] = ckpt['best_acc5']

        curr_and_best['curr_val_acc1'] = ckpt['curr_acc1']
        curr_and_best['curr_val_acc5'] = ckpt['curr_acc5']
    else:
        start_epoch = 0

    print("\n"+str(model)+"\n")
  
    if args.flips:
        print("Pruning with flipping ...\n")
        next_flip_epoch = args.flips[0]
        flip_idx = 1
   
    if args.pruning_strategy is None:
        print("Training via SGD\n")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
        scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)
    elif args.pruning_strategy == "activations":
        print("Pruning activations\n")
        prune = prune_activations
    elif args.pruning_strategy == "simulated_annealing":
        print("Pruning weights via simulated annealing\n")
        prune = simulated_annealing
    elif args.pruning_strategy == "weights":
        print("Pruning weights\n")
        prune = prune_weights
    elif args.pruning_strategy == "activations_and_weights":
        print("Pruning activations and then weights")
        print("Pruning activations ...\n")
        prune = prune_activations
    else:
        print("Please correctly specify the pruning-strategy argument. Can be \"weights\", \"activations\", \"simulated_annealing\", or None.")
        print("Exiting ...")
        sys.exit()

    train_time = 0
    start = time.time()
    update_curr_and_best_results(curr_and_best, model, device, data, criterion, args.batch_size)

    if args.save_plot_data:
        plot_data = {
            'train_acc1': [curr_and_best['curr_train_acc1']],
            'val_acc1': [curr_and_best['curr_val_acc1']],
            'epoch' : [0]
        }

    save_checkpoint(    # save initial state
        {
            'epoch': 0,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': curr_and_best['best_val_acc1'],
            'best_acc5': curr_and_best['best_val_acc5'],
            'best_train_acc1': curr_and_best['best_train_acc1'],
            'best_train_acc5': curr_and_best['best_train_acc5'],
            'optimizer': optimizer.state_dict() if 'optimizer' in locals() or 'optimizer' in globals() else None,
            'curr_acc1': curr_and_best['curr_val_acc1'],
            'curr_acc5': curr_and_best['curr_val_acc5'],
        },
        False,
        filename=ckpt_dir / f"initial.state",
        save=False,
    )

    for epoch in range(start_epoch, args.num_epochs):
        print("\nEpoch:", epoch+1)
            
        if args.flips:
            if epoch == next_flip_epoch:
                print("Flipping 10% of weights randomly ...")
                flip(model)
                if flip_idx < len(args.flips):
                    next_flip_epoch = args.flips[flip_idx]
                    flip_idx += 1

        start_train = time.time()
        
        if args.pruning_strategy == None:
            train(model, device, data.train_loader, optimizer, criterion, epoch+1, args.log_interval)
            scheduler.step()
        else:
            _, _, hamming_dist = prune(model, device, data.train_loader, criterion, args)
            print("\nHamming distance between current and previous masks: {}".format(hamming_dist))
        
        end_train = time.time()
        train_time += (end_train - start_train) / 3600      # in hours

        print("Network Sparsity: {:.2f}%".format(get_sparsity(model) * 100))
        is_best = update_curr_and_best_results(curr_and_best, model, device, data, criterion, args.batch_size)

        if args.save_plot_data:
            plot_data['train_acc1'].append(curr_and_best['curr_train_acc1'])
            plot_data['val_acc1'].append(curr_and_best['curr_val_acc1'])
            plot_data['epoch'].append(epoch + 1)

        save = (((epoch+1) % args.ckpt_interval) == 0) and args.ckpt_interval > 0
        if is_best or save or (epoch == (args.num_epochs - 1)):
            if is_best:
                print(f"New best! Saving to {ckpt_dir / 'best_model.state'}")

            save_checkpoint(
                {
                    'epoch': epoch+1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict() if 'optimizer' in locals() or 'optimizer' in globals() else None,
                    'sparsity': get_sparsity(model) * 100,
                    'best_acc1': curr_and_best['best_val_acc1'],
                    'best_acc5': curr_and_best['best_val_acc5'],
                    'best_train_acc1': curr_and_best['best_train_acc1'],
                    'best_train_acc5': curr_and_best['best_train_acc5'],
                    'curr_acc1': curr_and_best['curr_val_acc1'],
                    'curr_acc5': curr_and_best['curr_val_acc5']
                },
                is_best,
                filename=ckpt_dir / f"epoch_{epoch+1}.state",
                save=save,
            )

    if args.pruning_strategy == "activations_and_weights":
        num_children = len([i for i in model.children()])
        for idx, child in enumerate(model.children()):
            if idx < num_children - 1:      # if activations in the output layer are pruned, add the ability to reconnect them to the previous layer
                for i in range(len(child.mask_weight)):
                    if (child.mask_weight[i] == 0).all():
                        child.pruned_activation[i] = True

        if args.flips:
            next_flip_epoch = args.flips[0]
            flip_idx = 1

        print("\nPruning weights ...")
        for epoch in range(epoch + 1, args.num_epochs + epoch + 1):
            print("\nEpoch:", epoch+1)
            
            if args.flips:
                if (epoch - args.num_epochs) == next_flip_epoch:
                    print("Flipping 10% of weights randomly ...")
                    flip(model)
                    if flip_idx < len(args.flips):
                        next_flip_epoch = args.flips[flip_idx]
                        flip_idx += 1

            start_train = time.time()
            
            _, _, hamming_dist = prune_weights(model, device, data.train_loader, criterion, args)
            print("\nHamming distance between current and previous masks: {}".format(hamming_dist))
            
            end_train = time.time()
            train_time += (end_train - start_train) / 3600      # in hours

            print("Network Sparsity: {:.2f}%".format(get_sparsity(model) * 100))
            is_best = update_curr_and_best_results(curr_and_best, model, device, data, criterion, args.batch_size)

            if args.save_plot_data:
                plot_data['train_acc1'].append(curr_and_best['curr_train_acc1'])
                plot_data['val_acc1'].append(curr_and_best['curr_val_acc1'])
                plot_data['epoch'].append(epoch + 1)

            save = (((epoch+1) % args.ckpt_interval) == 0) and args.ckpt_interval > 0
            if is_best or save or (epoch == (args.num_epochs - 1)):
                if is_best:
                    print(f"New best! Saving to {ckpt_dir / 'best_model.state'}")

                save_checkpoint(
                    {
                        'epoch': epoch+1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict() if 'optimizer' in locals() or 'optimizer' in globals() else None,
                        'sparsity': get_sparsity(model) * 100,
                        'best_acc1': curr_and_best['best_val_acc1'],
                        'best_acc5': curr_and_best['best_val_acc5'],
                        'best_train_acc1': curr_and_best['best_train_acc1'],
                        'best_train_acc5': curr_and_best['best_train_acc5'],
                        'curr_acc1': curr_and_best['curr_val_acc1'],
                        'curr_acc5': curr_and_best['curr_val_acc5']
                    },
                    is_best,
                    filename=ckpt_dir / f"epoch_{epoch+1}.state",
                    save=save,
                )
    
    end = time.time()
    total_time = (end - start) / 3600    # in hours

    print("\nTotal train time: {:.0f} hr {:.0f} min {:.0f} sec".format(int(train_time), (train_time % 1) * 60, (((train_time % 1) * 60) % 1) * 60))
    print("Total train/test time: {:.0f} hr {:.0f} min {:.0f} sec".format(int(total_time), (total_time % 1) * 60, (((total_time % 1) * 60) % 1) * 60))
    print("The network is {:.2f} % sparse".format(get_sparsity(model) * 100))

    write_result_to_csv(
        best_val_acc1=curr_and_best['best_val_acc1'],
        best_val_acc5=curr_and_best['best_val_acc5'],
        best_train_acc1=curr_and_best['best_train_acc1'],
        best_train_acc5=curr_and_best['best_train_acc5'],
        curr_val_acc1=curr_and_best['curr_val_acc1'],
        curr_val_acc5=curr_and_best['curr_val_acc5'],
        train_time=train_time,
        total_time=total_time,
        sparsity=get_sparsity(model) * 100,
        base_config=args.config,
        name=args.name
    )


    if args.save_plot_data:
        pd.DataFrame(plot_data).to_csv(base_dir / "plot_data.csv", index=False)

#        with open(base_dir / "plot_data.csv", 'w') as csvfile:
#            writer = csv.DictWriter(csvfile, fieldnames=list(plot_data.keys()))
#            writer.writeheader()
#            for item in plot_data:
#                writer.writerow(item)

    print("\nExperiment complete! Exiting ...")


def get_device(args):
    if args.gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cuda:" + str(args.gpu)
  
    if torch.cuda.is_available():
        torch.cuda.device(device)

    print("Using device {} for training and testing".format(device))

    return device


def get_dataset(args):
    print("Benchmarking with the {} dataset".format(args.dataset))
    dataset = getattr(data, args.dataset.upper())(args)
    
    return dataset


def get_model(args, data, device):
    print("Creating model {}".format(args.arch))
    model = models.__dict__[args.arch](data.INPUT_SIZE, data.NUM_CLASSES, args)
    
    if not device == "cpu":
        model.cuda(device)
    
    if args.freeze_weights:
        freeze_model_weights(model)

    if args.start_from_nothing:             # TODO: get this working for convolutional layers!
        for child_idx, child in enumerate(model.children()):
            if hasattr(child, "mask_weight"):
                for i in range(len(child.mask_weight)):
                    child.mask_weight[i] = 0

                if child_idx == 0:
                    weight_idx_in = random.choice(range(len(child.weight[0])))

                weight_idx_out = random.choice(range(len(child.weight))),
                child.mask_weight[weight_idx_out, weight_idx_in] = 1
                print(torch.sum(child.mask_weight))

                weight_idx_in = weight_idx_out

    return model


def freeze_model_weights(model):
    print("\nFreezing model weights:")

    for name, param in model.named_parameters():
        print("{:<40}".format(f"  (1) No gradient to {name}") + f"(2) Setting gradient of {name} to None")
        param.requires_grad = False
        param.grad = None

    print()


def unfreeze_model_weights(model):
    print("\nUnfreezing model weights:")

    for name, param in model.named_parameters():
        print(f"  Gradient to {name}")
        param.requires_grad = False

    print()


def update_curr_and_best_results(curr_and_best, model, device, data, criterion, batch_size):
    _, acc = inference(model, device, data.train_loader, data.NUM_CLASSES, criterion, batch_size, (1, 5), "Train set")
    curr_train_acc1 = acc[0]
    curr_train_acc5 = acc[1]
    
    _, acc = inference(model, device, data.test_loader, data.NUM_CLASSES, criterion, batch_size, (1, 5), "Val set")
    curr_val_acc1 = acc[0]
    curr_val_acc5 = acc[1]

    is_best = True if curr_val_acc1 > curr_and_best['best_val_acc1'] else False

    curr_and_best['best_train_acc1'] = max(curr_train_acc1, curr_and_best['best_train_acc1'])
    curr_and_best['best_train_acc5'] = max(curr_train_acc5, curr_and_best['best_train_acc5'])
    curr_and_best['best_val_acc1'] = max(curr_val_acc1, curr_and_best['best_val_acc1'])
    curr_and_best['best_val_acc5'] = max(curr_val_acc5, curr_and_best['best_val_acc5'])

    curr_and_best['curr_train_acc1'] = curr_train_acc1
    curr_and_best['curr_train_acc5'] = curr_train_acc5
    curr_and_best['curr_val_acc1'] = curr_val_acc1
    curr_and_best['curr_val_acc5'] = curr_val_acc5

    return is_best
    

def write_result_to_csv(**kwargs):
    results = pathlib.Path("./results/greedy") / (args.arch + "_results.csv")

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Total Train Time (hrs), "
            "Total Time (hrs), "
            "Sparsity, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{train_time:.2f}, "
                "{total_time:.2f}, "
                "{sparsity:.2f}, "
                "{curr_val_acc1:.2f}, "
                "{curr_val_acc5:.2f}, "
                "{best_val_acc1:.2f}, "
                "{best_val_acc5:.2f}, "
                "{best_train_acc1:.2f}, "
                "{best_train_acc5:.2f}\n"
            ).format(now=now, **kwargs)
        )


def save_checkpoint(state, is_best, filename, save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "best_model.state"))

        if not save:
            os.remove(filename)


if __name__ == "__main__":
    main()
