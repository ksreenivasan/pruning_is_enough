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
        kwargs = {}  # {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")
        testdir = os.path.join(data_root, "test")

        '''
        if False: #True:
            normalize = transforms.Normalize(
                mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]
            )

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomRotation(20),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        '''
        if True:
            normalize = transforms.Normalize(
                mean=[0.48024578664982126,
                      0.44807218089384643, 0.3975477478649648],
                std=[0.2769864069088257, 0.26906448510256, 0.282081906210584]
            )

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomCrop(64, padding=4),
                        transforms.RandomHorizontalFlip(),
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

        self.actual_val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
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
