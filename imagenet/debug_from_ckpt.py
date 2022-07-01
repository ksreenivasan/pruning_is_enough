from imagenet_main import *

# do DDP stuff so I can load checkpoints
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:2500', world_size=1, rank=0)
model = models.WideResNet50_2()
gpu = 0
torch.cuda.set_device(gpu)
model.cuda(gpu)

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler.step()

arch = "WideResNet50_2"
ckpt3 = torch.load("results_ep_loglr/model_before_finetune_epoch_3.pth")
ckpt4 = torch.load("results_ep_loglr/model_before_finetune_epoch_4.pth")
ckpt5 = torch.load("results_ep/model_before_finetune_epoch_5.pth")
ckpt6 = torch.load("results_ep/model_before_finetune_epoch_6.pth")
ckpt11 = torch.load("results_ep/model_before_finetune_epoch_11.pth")

device = torch.device("cuda:{}".format(0))
model.load_state_dict(ckpt3)

cudnn.benchmark = True
args_data = "/data/imagenet"

# Data loading code
traindir = os.path.join(args_data, 'train')
valdir = os.path.join(args_data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

val_size = 10000
train_size = len(dataset) - val_size
train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

train_sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=(train_sampler is None),
    num_workers=12, pin_memory=True, sampler=train_sampler)

actual_val_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=128, shuffle=(train_sampler is None),
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

criterion = nn.CrossEntropyLoss().cuda(gpu)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg

validate(val_loader, model, criterion)

# returns num_nonzero elements, total_num_elements so that it is easier to compute
# average sparsity in the end
def get_layer_sparsity(layer, threshold=0, args=None):
    # assume the model is rounded, compute effective scores
    eff_scores, eff_bias_scores = models.GetSubnet.apply(layer.scores, layer.bias_scores, 0.5)
    if False:
        eff_bias_scores = layer.bias_scores * layer.bias_flag
    num_middle = torch.sum(torch.gt(eff_scores,
                           torch.ones_like(eff_scores)*threshold) *
                           torch.lt(eff_scores,
                           torch.ones_like(eff_scores.detach()*(1-threshold)).int()))
    if num_middle > 0:
        print("WARNING: Model scores are not binary. Sparsity number is unreliable.")
        raise ValueError
    w_numer, w_denom = eff_scores.detach().sum().item(), eff_scores.detach().flatten().numel()

    if False:
        b_numer, b_denom = eff_bias_scores.detach().sum().item(), eff_bias_scores.detach().flatten().numel()
    else:
        b_numer, b_denom = 0, 0

    return w_numer, w_denom, b_numer, b_denom


# returns avg_sparsity = number of non-zero weights!
def get_model_sparsity(model, threshold=0, args=None):
    conv_layers, linear_layers = get_layers(arch, model)
    numer = 0
    denom = 0

    # TODO: find a nicer way to do this (skip dropout)
    # TODO: Update: can't use .children() or .named_modules() because of the way things are wrapped in builder
    for conv_layer in conv_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(
            conv_layer, threshold, args=None)
        numer += w_numer
        denom += w_denom
        if False:
            numer += b_numer
            denom += b_denom

    for lin_layer in linear_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(
            lin_layer, threshold, args=None)
        numer += w_numer
        denom += w_denom
        if False:
            numer += b_numer
            denom += b_denom
    # print('Overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))
    return 100*numer/denom

avg_sparsity = get_model_sparsity(model, threshold=0, args=None)

model.load_state_dict(ckpt4)
acc4 = validate(val_loader, model, criterion)

model.load_state_dict(ckpt5)
acc5 = validate(val_loader, model, criterion)

model.load_state_dict(ckpt6)
acc6 = validate(val_loader, model, criterion)

model.load_state_dict(ckpt11)
acc11 = validate(val_loader, model, criterion)