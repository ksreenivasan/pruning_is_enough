from imagenet_main import *

# do DDP stuff so I can load checkpoints
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:2500', world_size=1, rank=0)
model = models.ResNet50()
gpu = 0
torch.cuda.set_device(gpu)
model.cuda(gpu)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
ckpt = torch.load("gm_results_latest/model_before_fineune_epoch_29.pth")

device = torch.device("cuda:{}".format(0))
model.load_state_dict(ckpt)

# make things regular again
model = model.module

cudnn.benchmark = True

# Data loading code
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=(train_sampler is None),
    num_workers=12, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False,
    num_workers=12, pin_memory=True)

