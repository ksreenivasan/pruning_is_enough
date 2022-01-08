# Follwed data preprocessing in https://github.com/snu-mllab/PuzzleMix/blob/master/load_data.py

import os

import torch
from torchvision import datasets, transforms

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

class TinyImageNet:
    def __init__(self, args):
        super(TinyImageNet, self).__init__()

        #data_root = os.path.join("tiny-imagenet-200")
        data_root = os.path.join(args.data, "tiny-imagenet-200")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {} # {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")
        testdir = os.path.join(data_root, "test")

        normalize = transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]
            #mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomCrop(64, padding=4),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                testdir, #valdir, #TODO: change here
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )

        self.test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                testdir,
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
