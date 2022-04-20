import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


def one_batch_dataset(dataset, batch_size):
    print("==> Grabbing a single batch")

    perm = torch.randperm(len(dataset))

    one_batch = [dataset[idx.item()] for idx in perm[:batch_size]]

    class _OneBatchWrapper(Dataset):
        def __init__(self):
            self.batch = one_batch

        def __getitem__(self, index):
            return self.batch[index]

        def __len__(self):
            return len(self.batch)

    return _OneBatchWrapper()


# used to build data transformation for ViT
def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
