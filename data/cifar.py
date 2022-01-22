import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args_helper import parser_args

from typing import List

import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter


class CIFAR10:
    def __init__(self, args):
        super(CIFAR10, self).__init__()


    data_root = os.path.join(parser_args.data, "cifar10")
    use_cuda = torch.cuda.is_available()
    # Data loading code
    kwargs = {"num_workers": parser_args.workers, "pin_memory": True} if use_cuda else {}

    datasets = {
    'train': torchvision.datasets.CIFAR10('/tmp', train=True, download=True),
    'test': torchvision.datasets.CIFAR10('/tmp', train=False, download=True)
    }

    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'/tmp/cifar_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    BATCH_SIZE = 512

    loaders = {}
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Create loaders
        loaders[name] = Loader(f'/tmp/cifar_{name}.beton',
                                batch_size=BATCH_SIZE,
                                num_workers=8,
                                order=OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline,
                                           'label': label_pipeline})

    self.train_loader = loaders['train']
    self.val_loader = loaders['test']
