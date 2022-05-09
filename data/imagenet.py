import os

import torch
from torchvision import datasets, transforms
import torch.multiprocessing
from args_helper import parser_args
from torch.utils.data import random_split

torch.multiprocessing.set_sharing_strategy("file_system")

class ImageNet:
    def __init__(self, args):
        super(ImageNet, self).__init__()

        data_root = parser_args.data

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": parser_args.num_workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        test_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

        if parser_args.use_full_data:
            train_dataset = dataset
            # use_full_data => we are not tuning hyperparameters
            validation_dataset = test_dataset
        else:
            train_size = 1000
            val_size = len(dataset) - train_size
            train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

        if parser_args.multiprocessing_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None


        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=parser_args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=parser_args.batch_size,
            shuffle=True,
            **kwargs
        )

        self.actual_val_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=parser_args.batch_size,
            shuffle=True,
            **kwargs
        )
