from args_helper import parser_args
import numpy as np
import torch

def get_scheduler(optimizer, policy='multistep_lr', milestones=[80, 120], gamma=0.1, max_epochs=150):

    if parser_args.epochs in [6]:
        milestones = [3,]
        gamma = parser_args.lr_gamma
    elif parser_args.epochs == 100:
        milestones = [50, 80]
        max_epochs = 100
    elif parser_args.epochs in [150, 160]:
        milestones = [80, 120]
        max_epochs = parser_args.epochs
    elif parser_args.epochs == 200:
        milestones = [100, 150]
        max_epochs = 200
    elif parser_args.epochs == 300:
        milestones = [150, 250]
        max_epochs = 300
    else:
        max_epochs = parser_args.epochs
        milestones = [int(max_epochs / 2.), int(max_epochs / 3. * 2.)]

    if policy == 'multistep_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        print("scheduler: use multistep learning rate decay, with milestones {} and gamma {}".format(milestones, gamma))

    elif policy == 'cosine_lr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        print("scheduler: use cosine learning rate decay, with max epochs {}".format(max_epochs))
    
    else:
        print("Policy not specified. Default is constant LR")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=1)
        #scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1)


    return scheduler


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
