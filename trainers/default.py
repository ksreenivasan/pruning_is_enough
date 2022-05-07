import time
import torch
# import tqdm
import copy
import pdb

from utils.eval_utils import accuracy
# from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import get_regularization_loss, prune, get_layers

from torch import optim
import psutil

__all__ = ["train", "validate", "modifier"]



def train(train_loader, model, criterion, optimizer, epoch, args, writer, scaler=None):
    # batch_time = AverageMeter("Time", ":6.3f")
    # data_time = AverageMeter("Data", ":6.3f")
    # losses = AverageMeter("Loss", ":.3f")
    # top1 = AverageMeter("Acc@1", ":6.2f")
    # top5 = AverageMeter("Acc@5", ":6.2f")
    # top10 = AverageMeter("Acc@10", ":6.2f")
    # progress = ProgressMeter(
    #    len(train_loader),
    #    [batch_time, data_time, losses, top1, top5],
    #    prefix=f"GPU:[{args.gpu}] | Epoch: [{epoch}]",
    # )
    top1 = 0
    top5 = 0
    top10 = 0
    num_images = 0

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    # for i, (images, target) in tqdm.tqdm(
    #     enumerate(train_loader), ascii=True, total=len(train_loader)
    # ):
    if args.gpu == 0:
        print("(TRAINER)BEFORE TRAIN LOOP: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time = time.time() - end
        # print("Data Time: {}".format(data_time))
        # print(images.shape, target.shape)

        images = images.to(args.gpu)
        target = target.to(args.gpu)

        # update score thresholds for global ep
        if args.algo in ['global_ep', 'global_ep_iter']:
            prune(model, update_thresholds_only=True)

        if args.algo in ['hc', 'hc_iter', 'pt'] and i % args.project_freq == 0 and not args.differentiate_clamp:
            for name, params in model.named_parameters():
                if "score" in name:
                    scores = params
                    with torch.no_grad():
                        scores.data = torch.clamp(scores.data, 0.0, 1.0)

        # compute output
        if scaler is None:
            output = model(images)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast(enabled=True): # mixed precision
                output = model(images)
                loss = criterion(output, target)

        if args.lam_finetune_loss > 0:
            raise NotImplementedError  # please check finetune_loss repo

        regularization_loss = torch.tensor(0)
        if args.gpu == 0:
            print("(TRAINER)BEFORE REGULARIZATION COMPUTATION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
        if args.regularization:
            regularization_loss =\
                get_regularization_loss(model, regularizer=args.regularization,
                                        lmbda=args.lmbda, alpha=args.alpha,
                                        alpha_prime=args.alpha_prime)
        if args.gpu == 0:
            print("(TRAINER)BEFORE REGULARIZATION COMPUTATION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
        #print('regularization_loss: ', regularization_loss)
        loss += regularization_loss
        if args.gpu == 0:
            print("(TRAINER)BEFORE ACCURACY COMPUTATION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
        # measure accuracy and record loss
        acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
        # losses.update(loss.item(), images.size(0))
        # top1.update(acc1.item(), images.size(0))
        # top5.update(acc5.item(), images.size(0))
        # top10.update(acc10.item(), images.size(0))
        # compute weighted sum for each accuracy so we can average it later
        top1 += acc1.item() * images.size(0)
        top5 += acc5.item() * images.size(0)
        top10 += acc10.item() * images.size(0)
        num_images += images.size(0)

        if args.gpu == 0:
            print("(TRAINER)AFTER ACCURACY COMPUTATION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        batch_time = time.time() - end
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            # progress.display(i)
            # progress.write_to_tensorboard(
            #     writer, prefix="train", global_step=t)
            print("GPU:{} | Epoch: {} | loss={} | Batch Time={}".format(args.gpu, epoch, loss.item(), acc1.item(), batch_time))

    print("(TRAINER)AFTER TRAIN LOOP: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
    # before completing training, clean up model based on latest scores
    # update score thresholds for global ep
    if args.algo in ['global_ep', 'global_ep_iter']:
        prune(model, update_thresholds_only=True)
    if args.algo in ['hc', 'hc_iter', 'pt'] and not args.differentiate_clamp:
        if args.gpu == 0:
            print("(TRAINER)BEFORE PROJECTION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
        for name, params in model.named_parameters():
            if "score" in name:
                scores = params
                with torch.no_grad():
                    scores.data = torch.clamp(scores.data, 0.0, 1.0)
        if args.gpu == 0:
            print("(TRAINER)AFTER PROJECTION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))

    return top1/num_images, top5/num_images, top10/num_images, regularization_loss.item()


def validate(val_loader, model, criterion, args, writer, epoch):
    # batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    # losses = AverageMeter("Loss", ":.3f", write_val=False)
    # top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    # top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    # top10 = AverageMeter("Acc@10", ":6.2f", write_val=False)
    # progress = ProgressMeter(
    #    len(val_loader), [batch_time, losses, top1, top5, top10], prefix="Test: "
    # )
    top1 = 0
    top5 = 0
    top10 = 0
    num_images = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        # for i, (images, target) in tqdm.tqdm(
        #     enumerate(val_loader), ascii=True, total=len(val_loader)
        # ):
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            #print(images.shape, target.shape)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
            # losses.update(loss.item(), images.size(0))
            # top1.update(acc1.item(), images.size(0))
            # top5.update(acc5.item(), images.size(0))
            # top10.update(acc10.item(), images.size(0))
            # compute weighted sum for each accuracy so we can average it later
            top1 += acc1.item() * images.size(0)
            top5 += acc5.item() * images.size(0)
            top10 += acc10.item() * images.size(0)
            num_images += images.size(0)

            # measure elapsed time
            # batch_time.update(time.time() - end)
            batch_time = time.time() - end
            end = time.time()

            if i % args.print_freq == 0:
                # progress.display(i)
                print("GPU:{} | Epoch: {} | loss={} | Batch Time={}".format(args.gpu, epoch, loss.item(), acc1.item(), batch_time))

        # progress.display(len(val_loader))

        # if writer is not None:
        #     progress.write_to_tensorboard(
        #         writer, prefix="test", global_step=epoch)

    print("Model top1 Accuracy: {}".format(top1/num_images))
    return top1/num_images, top5/num_images, top10/num_images


def modifier(args, epoch, model):
    return
