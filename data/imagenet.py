import os

import torch
from torchvision import datasets, transforms

import torch.multiprocessing

from .utils import build_transform

torch.multiprocessing.set_sharing_strategy("file_system")

class ImageNet:
    def __init__(self, args):
        super(ImageNet, self).__init__()

#        data_root = os.path.join(args.data, "imagenet")
        data_root = args.data

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.num_workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if 'deit' in args.arch:     # special transform for ViT models
            train_transform = build_transform(True, args)
            val_transform = build_transform(False, args)
            drop_last = True
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            val_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            drop_last = False

        train_dataset = datasets.ImageFolder(traindir, transform=train_transform)

        if args.multiprocessing_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None


        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=drop_last,
            **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir, transform=val_transform
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
