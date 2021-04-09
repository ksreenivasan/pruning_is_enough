import os
import pathlib
import random
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
)
from utils.schedulers import get_policy


from args import args as parser_args
import importlib

import data
import models


def main():
    print(parser_args)

    if parser_args.seed is not None:
        random.seed(parser_args.seed)
        torch.manual_seed(parser_args.seed)
        torch.cuda.manual_seed(parser_args.seed)
        torch.cuda.manual_seed_all(parser_args.seed)

    # Simply call main_worker function
    main_worker(parser_args)


def main_worker(parser_args):
    parser_args.gpu = None
    train, validate, modifier = get_trainer(parser_args)

    if parser_args.gpu is not None:
        print("Use GPU: {} for training".format(parser_args.gpu))

    # create model and optimizer
    model = get_model(parser_args)
    model = set_gpu(parser_args, model)

    if parser_args.pretrained:
        pretrained(parser_args, model)

    optimizer = get_optimizer(parser_args, model)
    data = get_dataset(parser_args)
    lr_policy = get_policy(parser_args.lr_policy)(optimizer, parser_args)

    if parser_args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=parser_args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if parser_args.resume:
        best_acc1 = resume(parser_args, model, optimizer)

    # Data loading code
    if parser_args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, parser_args, writer=None, epoch=parser_args.start_epoch
        )

        return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(parser_args)
    parser_args.ckpt_base_dir = ckpt_base_dir

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    parser_args.start_epoch = parser_args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": parser_args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    # Start training
    for epoch in range(parser_args.start_epoch, parser_args.epochs):
        lr_policy(epoch, iteration=None)
        modifier(parser_args, epoch, model)

        cur_lr = get_lr(optimizer)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, parser_args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, parser_args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % parser_args.save_every) == 0) and parser_args.save_every > 0
        if is_best or save or epoch == parser_args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": parser_args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        if parser_args.conv_type == "SampleSubnetConv":
            count = 0
            sum_pr = 0.0
            for n, m in model.named_modules():
                if isinstance(m, SampleSubnetConv):
                    # avg pr across 10 samples
                    pr = 0.0
                    for _ in range(10):
                        pr += (
                            (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
                            .float()
                            .mean()
                            .item()
                        )
                    pr /= 10.0
                    writer.add_scalar("pr/{}".format(n), pr, epoch)
                    sum_pr += pr
                    count += 1

            parser_args.prune_rate = sum_pr / count
            writer.add_scalar("pr/average", parser_args.prune_rate, epoch)

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=parser_args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=parser_args.config,
        name=parser_args.name,
    )


def get_trainer(parser_args):
    print(f"=> Using trainer from trainers.{parser_args.trainer}")
    trainer = importlib.import_module(f"trainers.{parser_args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(parser_args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if parser_args.gpu is not None:
        torch.cuda.set_device(parser_args.gpu)
        model = model.cuda(parser_args.gpu)
    elif parser_args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {parser_args.multigpu} gpus")
        torch.cuda.set_device(parser_args.multigpu[0])
        parser_args.gpu = parser_args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=parser_args.multigpu).cuda(
            parser_args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume(parser_args, model, optimizer):
    if os.path.isfile(parser_args.resume):
        print(f"=> Loading checkpoint '{parser_args.resume}'")

        checkpoint = torch.load(parser_args.resume, map_location=f"cuda:{parser_args.multigpu[0]}")
        if parser_args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            parser_args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{parser_args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{parser_args.resume}'")


def pretrained(parser_args, model):
    if os.path.isfile(parser_args.pretrained):
        print("=> loading pretrained weights from '{}'".format(parser_args.pretrained))
        pretrained = torch.load(
            parser_args.pretrained,
            map_location=torch.device("cuda:{}".format(parser_args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(parser_args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


def get_dataset(parser_args):
    print(f"=> Getting {parser_args.dataset} dataset")
    dataset = getattr(data, parser_args.dataset)(parser_args)

    return dataset


def get_model(parser_args):
    if parser_args.first_layer_dense:
        parser_args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(parser_args.arch))
    model = models.__dict__[parser_args.arch]()

    # applying sparsity to the network
    if (
        parser_args.conv_type != "DenseConv"
        and parser_args.conv_type != "SampleSubnetConv"
        and parser_args.conv_type != "ContinuousSparseConv"
    ):
        if parser_args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

        set_model_prune_rate(model, prune_rate=parser_args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-parser_args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    if parser_args.freeze_weights:
        freeze_model_weights(model)

    return model


def get_optimizer(parser_args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if parser_args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if parser_args.no_bn_decay else parser_args.weight_decay,
                },
                {"params": rest_params, "weight_decay": parser_args.weight_decay},
            ],
            parser_args.lr,
            momentum=parser_args.momentum,
            weight_decay=parser_args.weight_decay,
            nesterov=parser_args.nesterov,
        )
    elif parser_args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=parser_args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(parser_args):
    if parser_args.config is None or parser_args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(parser_args.config).stem
    if parser_args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{parser_args.name}/prune_rate={parser_args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{parser_args.log_dir}/{config}/{parser_args.name}/prune_rate={parser_args.prune_rate}"
        )
    if parser_args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(parser_args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(parser_args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()