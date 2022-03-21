"""
    script to create ffcv_imagenet train loader
"""

import os
import torch
import torchvision
from torchvision import datasets, transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args_helper import parser_args
import numpy as np
from pathlib import Path
from typing import List

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

class FfcvImageNet:
    def __init__(self, args):
        super(FfcvImageNet, self).__init__()

        # data_root = os.path.join(parser_args.data, "imagenet")
        # put ffcv path here
        data_root = parser_args.data

        use_cuda = torch.cuda.is_available()

        # Data loading code
        train_dataset = os.path.join(data_root, "train_500_0.50_90.ffcv")
        val_dataset = os.path.join(data_root, "val_500_0.50_90.ffcv")

        # Data loading code
        kwargs = {"num_workers": 1, "in_memory": 1,
                  "distributed": False, "resolution": 256}

        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
        DEFAULT_CROP_RATIO = 224/256

        self.train_loader = self.create_train_loader(train_dataset,
                                kwargs['num_workers'],
                                parser_args.batch_size,
                                kwargs['distributed'],
                                kwargs['in_memory'])
        self.val_loader = self.create_val_loader(val_dataset,
                                kwargs['num_workers'],
                                parser_args.batch_size,
                                kwargs['resolution'],
                                kwargs['distributed'],
                                )
        # madry does this but I don't think we need to
        # self.model, self.scaler = self.create_model_and_scaler()

    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory):
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(parser_args.gpu), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(parser_args.gpu), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader


    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(parser_args.gpu), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(parser_args.gpu),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    def get_resolution(self, epoch=0, min_res=160, max_res=192, end_ramp=76, start_ramp=65):
        # this seems to be a hack to get good accuracy and is only between epochs
        # 65 and 76. So, for now just return max_res. always.

        return max_res

        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res
