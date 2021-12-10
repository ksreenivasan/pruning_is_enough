import time
import torch
import tqdm
import copy
import pdb

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import get_regularization_loss, prune, get_layers

from torch import optim

__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    top10 = AverageMeter("Acc@10", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # update score thresholds for global ep
        if args.algo in ['global_ep', 'global_ep_iter']:
            prune(model, update_thresholds_only=True)

        # compute output
        output = model(images)

        if args.algo in ['hc', 'hc_iter', 'pt'] and i % args.project_freq == 0 and not args.differentiate_clamp:
            for name, params in model.named_parameters():
                if "score" in name:
                    scores = params
                    with torch.no_grad():
                        scores.data = torch.clamp(scores.data, 0.0, 1.0)

        loss = criterion(output, target)


        if args.lam_finetune_loss > 0:

            model2 = copy.deepcopy(model)
            #delta_list = {}
            conv, linear = get_layers(arch=args.arch, model=model)
            conv2, linear2 = get_layers(arch=args.arch, model=model2)
            layer_list = [*conv, *linear]
            layer_list2 = [*conv2, *linear2]

            #pdb.set_trace()
            # clone parameters from model
            for m_from, m_to in zip(layer_list, layer_list2):#model.modules(), model2.modules()):
                m_to.scores.data = m_from.scores.data.clone()
                m_to.flag.data = m_from.flag.data.clone()
            
            # turn on gradient for weight, turn off gradient for mask
            for i, (name, params) in enumerate(model2.named_parameters()):
                if "weight" in name:
                    params.requires_grad=True
                elif "score" in name:
                    params.requires_grad=False
                
            meta_optimizer = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
            # update for several steps
            num_updates = 10
            for i in range(num_updates):
                # forward and backward to update net_pi grad.
                loss_updated_model = criterion(model2(images), target)
                meta_optimizer.zero_grad()
                loss_updated_model.backward()#retain_graph=True)
                meta_optimizer.step()

            loss_updated_model = criterion(model2(images), target)
    

            '''
            # define updates
            for i, (name, params) in enumerate(model2.named_parameters()):
                if "weight" in name:
                    delta_list[i] = torch.autograd.grad(loss, params)

            model3 = clone(model)
            for i, (name, params) in enumerate(model3.named_parameters()):
                if "weight" in name:
                    params -= alpha * delta_list[i]
            output3 = model3(images)
            '''

            finetune_loss = args.lam_finetune_loss * loss_updated_model 
            loss += finetune_loss

            # go back to original setting
            for i, (name, params) in enumerate(model.named_parameters()):
                if "weight" in name:
                    params.requires_grad=False
                elif "score" in name:
                    params.requires_grad=True



        regularization_loss = torch.tensor(0)
        if args.regularization:
            regularization_loss =\
                get_regularization_loss(model, regularizer=args.regularization,
                                        lmbda=args.lmbda, alpha=args.alpha,
                                        alpha_prime=args.alpha_prime)

        loss += regularization_loss

        # measure accuracy and record loss
        acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        top10.update(acc10.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    # before completing training, clean up model based on latest scores
    # update score thresholds for global ep
    if args.algo in ['global_ep', 'global_ep_iter']:
        prune(model, update_thresholds_only=True)
    if args.algo in ['hc', 'hc_iter', 'pt'] and not args.differentiate_clamp:
        for name, params in model.named_parameters():
            if "score" in name:
                scores = params
                with torch.no_grad():
                    scores.data = torch.clamp(scores.data, 0.0, 1.0)

    return top1.avg, top5.avg, top10.avg, regularization_loss.item()


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    top10 = AverageMeter("Acc@10", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5, top10], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            top10.update(acc10.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    print("Model top1 Accuracy: {}".format(top1.avg))
    return top1.avg, top5.avg, top10.avg

def modifier(args, epoch, model):
    return
