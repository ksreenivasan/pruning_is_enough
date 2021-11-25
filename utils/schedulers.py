import numpy as np
import torch

def get_scheduler(optimizer, policy='multistep_lr', milestones=[80, 120], gamma=0.1, max_epochs=150):


    if policy == 'multistep_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif policy == 'cosine_lr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    
    else:
        print("Policy not specified. Default is constant LR")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=1)
        #scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1)


    return scheduler


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
