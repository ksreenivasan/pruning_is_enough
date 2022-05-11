from main_utils import *
import psutil
from torchvision import datasets, transforms
import torch.multiprocessing
from torch.utils.data import random_split
# torch.multiprocessing.set_sharing_strategy("file_system")



def main():
    print(parser_args)
    print("\n\nBeginning of process.")
    print_time()
    set_seed(parser_args.seed * parser_args.trial_num)
    # set_seed(parser_args.seed + parser_args.trial_num - 1)

    # world size = ngpus_per_node since we are assuming single node
    ngpus_per_node = torch.cuda.device_count()

    # if parser_args.multiprocessing_distributed:
    #     # assert ngpus_per_node >= 2, f"Requires at least 2 GPUs to run, but got {ngpus_per_node}"
    #     mp.spawn(main_worker, args=(ngpus_per_node,), nprocs=ngpus_per_node, join=True)
    # else:
    #     # Simply call main_worker function
    #     main_worker(parser_args.gpu, ngpus_per_node)
    main_worker(parser_args.rank, ngpus_per_node)


def main_worker(rank, ngpus_per_node):
    # NOTE: gpu = rank in the multiprocessing setting
    parser_args.gpu = rank

    if parser_args.gpu is not None:
        print("Use GPU: {} for training".format(parser_args.gpu))

    if parser_args.multiprocessing_distributed:
        setup_distributed(parser_args.gpu, ngpus_per_node)
        # if using ddp, divide batch size per gpu
        parser_args.batch_size = int(parser_args.batch_size / ngpus_per_node)

    # train, validate, modifier = get_trainer(parser_args)
    #model = get_model(parser_args)

    if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
        idty_str = get_idty_str(parser_args)
        if parser_args.subfolder is not None:
            if not os.path.isdir('results/'):
                os.mkdir('results/')
            result_subroot = 'results/' + parser_args.subfolder + '/'
            if not os.path.isdir(result_subroot):
                os.mkdir(result_subroot)
            result_root = result_subroot + '/results_' + idty_str + '/'
        else:
            result_root = 'results/results_' + idty_str + '/'

        if not os.path.isdir(result_root):
            os.mkdir(result_root)
        print_model(model, parser_args)
    else:
        idty_str = get_idty_str(parser_args)
        result_root = 'results/results_' + idty_str + '/'


    if parser_args.weight_training:
        model = round_model(model, round_scheme="all_ones", noise=parser_args.noise,
                            ratio=parser_args.noise_ratio, rank=parser_args.gpu)
        model = switch_to_wt(model)

    model = set_gpu(parser_args, model)

    if parser_args.pretrained:
        pretrained(parser_args.pretrained, model)
    if parser_args.pretrained2:
        # model2.load_state_dict(torch.load(parser_args.pretrained2)['state_dict'])
        model2 = copy.deepcopy(model)
        pretrained(parser_args.pretrained2, model2)
    else:
        model2 = None

    optimizer = get_optimizer(parser_args, model)
    data = get_dataset(parser_args)
    scheduler = get_scheduler(optimizer, parser_args.lr_policy)
    # lr_policy = get_policy(parser_args.lr_policy)(optimizer, parser_args)
    if parser_args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothing(smoothing=parser_args.label_smoothing)
        # if isinstance(model, nn.parallel.DistributedDataParallel):
        #     model = model.module
    if parser_args.random_subnet: 
        test_random_subnet(model, data, criterion, parser_args, result_root, parser_args.smart_ratio) 
        return
        

    best_acc1, best_acc5, best_acc10, best_train_acc1, best_train_acc5, best_train_acc10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # optionally resume from a checkpoint
    if parser_args.resume:
        best_acc1 = resume(parser_args, model, optimizer)
    # when we only evaluate a pretrained model
    if parser_args.evaluate:
        evaluate_without_training(
            parser_args, model, model2, validate, data, criterion)
        return

    # Set up directories & setting
    run_base_dir, ckpt_base_dir, log_base_dir, writer, epoch_time, validation_time, train_time, progress_overall = get_settings(
        parser_args)
    end_epoch = time.time()
    parser_args.start_epoch = parser_args.start_epoch or 0
    acc1 = None
    epoch_list, test_acc_before_round_list, test_acc_list, reg_loss_list, model_sparsity_list, val_acc_list, train_acc_list = [], [], [], [], [], [], []

    # Save the initial model
    #torch.save(model.state_dict(), result_root + 'init_model.pth')

    # compute prune_rate to reach target_sparsity
    if not parser_args.override_prune_rate:
        parser_args.prune_rate = get_prune_rate(
            parser_args.target_sparsity, parser_args.iter_period)
        print("Setting prune_rate to {}".format(parser_args.prune_rate))
    else:
        print("Overriding prune_rate to {}".format(parser_args.prune_rate))
    #if parser_args.dataset == 'TinyImageNet':
    #    print_num_dataset(data)
    if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
        if not parser_args.weight_training:
            print_layers(parser_args, model)

    if parser_args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True) # mixed precision
    else:
        scaler = None

    if parser_args.only_sanity:
        dirs = os.listdir(parser_args.sanity_folder)
        for path in dirs:
            parser_args.results_root = parser_args.sanity_folder +'/'+ path +'/' 
            parser_args.resume =parser_args.results_root + '/model_before_finetune.pth'
            resume(parser_args, model, optimizer)
            
            do_sanity_checks(model, parser_args, data, criterion, epoch_list, test_acc_before_round_list,
                         test_acc_list, reg_loss_list, model_sparsity_list, parser_args.results_root)
            
            # cp_model = round_model(model, round_scheme="all_ones", noise=parser_args.noise,
            #            ratio=parser_args.noise_ratio, rank=parser_args.gpu)
            # print(get_model_sparsity(cp_model))
        return


    # Start training
    for epoch in range(parser_args.start_epoch, parser_args.epochs):
        print("STARTING TRAINING: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
        if parser_args.multiprocessing_distributed:
            data.train_loader.sampler.set_epoch(epoch)
        # lr_policy(epoch, iteration=None)
        # modifier(parser_args, epoch, model)
        # cur_lr = get_lr(optimizer)

        print("Skipping training, just gonna round")
        print("Before Round: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
        # cp_model = round_model(model, parser_args.round, noise=parser_args.noise,
                                #ratio=parser_args.noise_ratio, rank=parser_args.gpu)
        if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
            # print("Pretend to validate")
            # continue
            # do_something()
            my_validate(data.actual_val_loader, None, criterion, parser_args, writer, epoch)
            acc1 = -1
        else:
            acc1 = -1
        print("GPU: {} | acc1={}".format(parser_args.gpu, acc1))
        print("After Round: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
        dist.barrier()
        continue


        # save the score at the beginning of training epoch, so if we set parser.args.rewind_to_epoch to 0
        # that means we save the initialization of score
        if parser_args.rewind_score and parser_args.rewind_to_epoch == epoch:
            # if rewind the score, checkpoint the score when reach the desired epoch (rewind to iteration not yet implemented)
            with torch.no_grad():
                conv_layers, linear_layers = get_layers(
                    parser_args.arch, model)
                for layer in [*conv_layers, *linear_layers]:
                    layer.saved_score.data = layer.score.data


        # train for one epoch
        start_train = time.time()
        print("BEFORE TRAIN LOOP: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
        train_acc1, train_acc5, train_acc10, reg_loss = train(
            data.train_loader, model, criterion, optimizer, epoch, parser_args, writer=writer, scaler=scaler
        )
        print("AFTER TRAIN LOOP: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
        # train_time.update((time.time() - start_train) / 60)
        train_time = (time.time() - start_train) / 60

        scheduler.step()

        # evaluate on validation set
        if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
            start_validation = time.time()
            print("BEFORE VAL LOOP: GPU: {} | Epoch {} | Memory Usage: {}".format(parser_args.gpu, epoch, psutil.virtual_memory()))
            if parser_args.algo in ['hc', 'hc_iter']:
                br_acc1, br_acc5, br_acc10 = validate(
                    data.val_loader, model, criterion, parser_args, writer, epoch)  # before rounding
                print('Acc before rounding: {}'.format(br_acc1))
                acc_avg = 0
                for num_trial in range(parser_args.num_test):
                    cp_model = round_model(model, parser_args.round, noise=parser_args.noise,
                                           ratio=parser_args.noise_ratio, rank=parser_args.gpu)
                    acc1, acc5, acc10 = validate(
                        data.val_loader, cp_model, criterion, parser_args, writer, epoch)
                    acc_avg += acc1
                acc_avg /= parser_args.num_test
                acc1 = acc_avg
                print('Acc after rounding: {}'.format(acc1))
                val_acc1, val_acc5, val_acc10 = validate(
                        data.actual_val_loader, cp_model, criterion, parser_args, writer, epoch)
                print('Validation Acc after rounding: {}'.format(val_acc1))
            else:
                acc1, acc5, acc10 = validate(
                    data.val_loader, model, criterion, parser_args, writer, epoch)
                print('Acc: {}'.format(acc1))
            # validation_time.update((time.time() - start_validation) / 60)
            validation_time = (time.time() - start_validation) / 60
            print("AFTER VAL LOOP: GPU: {} | Epoch {} | Memory Usage: {}".format(parser_args.gpu, epoch, psutil.virtual_memory()))

        # prune the model every T_{prune} epochs
        if not parser_args.weight_training and parser_args.algo in ['hc_iter', 'global_ep_iter'] and epoch % (parser_args.iter_period) == 0 and epoch != 0:
            print("BEFORE PRUNE: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
            prune(model)
            print("AFTER PRUNE: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
            if parser_args.checkpoint_at_prune:
                if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
                    save_checkpoint_at_prune(model, parser_args)

        # get model sparsity
        # DDP_TODO: This part could be a problem. What if models get out of sync?
        if not parser_args.weight_training:
            if parser_args.bottom_k_on_forward:
                cp_model = copy.deepcopy(model)
                prune(cp_model, update_scores=True)
                avg_sparsity = get_model_sparsity(cp_model)
            elif parser_args.algo in ['hc', 'hc_iter']:
                # Round before checking sparsity
                if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
                    cp_model = round_model(model, parser_args.round, noise=parser_args.noise,
                                           ratio=parser_args.noise_ratio, rank=parser_args.gpu)
                    avg_sparsity = get_model_sparsity(cp_model)
                else:
                    avg_sparsity = -1
            else:
                avg_sparsity = get_model_sparsity(model)
        else:
            # haven't written a weight sparsity function yet
            avg_sparsity = -1

        if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
            print('Model avg sparsity: {}'.format(avg_sparsity))

        # if model has been "short-circuited", then no point in continuing training
        if avg_sparsity == 0:
            print("\n\n---------------------------------------------------------------------")
            print("WARNING: Model Sparsity = 0 => Entire network has been pruned!!!")
            print("EXITING and moving to Fine-tune")
            print("---------------------------------------------------------------------\n\n")
            # TODO: Hacky code. Doesn't always work. But quick and easy fix. Just prune all weights to target
            # sparsity, and then continue to finetune so that unflag can do stuff.
            parser_args.prune_rate = 1 - (parser_args.target_sparsity/100)
            prune(model)
            break

        # update all results lists
        if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
            epoch_list.append(epoch)
            if parser_args.algo in ['hc', 'hc_iter']:
                test_acc_before_round_list.append(br_acc1)
            else:
                # no before rounding for EP/weight training
                test_acc_before_round_list.append(-1)
            test_acc_list.append(acc1)
            val_acc_list.append(val_acc1)
            train_acc_list.append(train_acc1)
            reg_loss_list.append(reg_loss)
            model_sparsity_list.append(avg_sparsity)

            # epoch_time.update((time.time() - end_epoch) / 60)
            epoch_time = (time.time() - end_epoch) / 60
            # progress_overall.display(epoch)
            # progress_overall.write_to_tensorboard(
            #     writer, prefix="diagnostics", global_step=epoch
            # )
            print("GPU:{} | Epoch: {} | Acc={} | Epoch Time={}".format(parser_args.gpu, epoch, acc1, epoch_time))

            # writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()

            if parser_args.algo in ['hc', 'hc_iter']:
                results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc_before_rounding': test_acc_before_round_list,
                                          'test_acc': test_acc_list, 'val_acc': val_acc_list, 'train_acc': train_acc_list, 'regularization_loss': reg_loss_list, 'model_sparsity': model_sparsity_list})
            else:
                results_df = pd.DataFrame(
                    {'epoch': epoch_list, 'test_acc': test_acc_list, 'model_sparsity': model_sparsity_list})

            if parser_args.results_filename:
                results_filename = parser_args.results_filename
            else:
                results_filename = result_root + 'acc_and_sparsity.csv'
            print("Writing results into: {}".format(results_filename))
            results_df.to_csv(results_filename, index=False)
            print("AFTER TRAINING: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
    print("Local rank: {} | About to enter save model logic".format(parser_args.gpu))
    if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
        # save checkpoint before fine-tuning
        # torch.save(model.state_dict(), result_root + 'model_before_finetune.pth')

        print("\n\nHigh accuracy subnetwork found! Rest is just finetuning")
        print("Local rank: {}".format(parser_args.gpu))
        print_time()

    # finetune weights
    # DDP works surprisingly well with copy deepcopy. Might cause memory issues TODO
    if parser_args.multiprocessing_distributed:
        print("TORCH BARRIER: GPU:{}".format(parser_args.gpu))
        dist.barrier()
        print("CLEARED TORCH BARRIER: GPU:{}".format(parser_args.gpu))

    cp_model = copy.deepcopy(model)
    if not parser_args.skip_fine_tune:
        if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
            print("Beginning fine-tuning")
        cp_model = finetune(cp_model, parser_args, data, criterion, epoch_list,
                            test_acc_before_round_list, test_acc_list, val_acc_list, train_acc_list, reg_loss_list, model_sparsity_list, result_root)
        # print out the final acc
        if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
            eval_and_print(validate, data.val_loader, cp_model, criterion,
                           parser_args, writer=None, description='final model after finetuning')
            # save checkpoint after fine-tuning
            torch.save(cp_model.state_dict(), result_root + 'model_after_finetune.pth')
    else:
        if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
            print("Skipping finetuning!!!")

    if parser_args.multiprocessing_distributed:
        print("TORCH BARRIER: GPU:{}".format(parser_args.gpu))
        dist.barrier()
        print("CLEARED TORCH BARRIER: GPU:{}".format(parser_args.gpu))

    if not parser_args.skip_sanity_checks:
        do_sanity_checks(model, parser_args, data, criterion, epoch_list, test_acc_before_round_list,
                         test_acc_list, val_acc_list, train_acc_list, reg_loss_list, model_sparsity_list, result_root)
    else:
        if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
            print("Skipping sanity checks!!!")

    if (parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
        print("\n\nEnd of process. Exiting")
        print_time()

    if parser_args.multiprocessing_distributed:
        cleanup_distributed()


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
    print("(TRAINER)BEFORE TRAIN LOOP: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
    for i, (images, target) in enumerate(train_loader):
        # print("(TRAINER)AFTER LOADING IMAGES: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
        # print("Is it just the image?")
        # print("(TRAINER): Size of images: {}".format(sys.getsizeof(images)))
        # measure data loading time
        data_time = time.time() - end
        # print("Data Time: {}".format(data_time))
        # print(images.shape, target.shape)

        images = images.to(args.gpu)
        target = target.to(args.gpu)

        # update score thresholds for global ep
        if args.algo in ['global_ep', 'global_ep_iter']:
            prune(model, update_thresholds_only=True)

        # print("(TRAINER)BEFORE PROJECTION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
        if args.algo in ['hc', 'hc_iter', 'pt'] and i % args.project_freq == 0 and not args.differentiate_clamp:
            for name, params in model.named_parameters():
                if "score" in name:
                    scores = params
                    with torch.no_grad():
                        scores.data = torch.clamp(scores.data, 0.0, 1.0)
        # print("(TRAINER)BEFORE PROJECTION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))

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
        # print("(TRAINER)BEFORE REGULARIZATION COMPUTATION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
        if args.regularization:
            regularization_loss =\
                get_regularization_loss(model, regularizer=args.regularization,
                                        lmbda=args.lmbda, alpha=args.alpha,
                                        alpha_prime=args.alpha_prime)
        # print("(TRAINER)BEFORE REGULARIZATION COMPUTATION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
        #print('regularization_loss: ', regularization_loss)
        loss += regularization_loss
        # print("(TRAINER)BEFORE ACCURACY COMPUTATION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
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

        # print("(TRAINER)AFTER ACCURACY COMPUTATION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))

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
        # print("(TRAINER)BEFORE PROJECTION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))
        for name, params in model.named_parameters():
            if "score" in name:
                scores = params
                with torch.no_grad():
                    scores.data = torch.clamp(scores.data, 0.0, 1.0)
        # print("(TRAINER)AFTER PROJECTION: GPU:{} | Epoch {} | Memory Usage: {}".format(args.gpu, epoch, psutil.virtual_memory()))

    return top1/num_images, top5/num_images, top10/num_images, regularization_loss.item()


def my_validate(val_loader, model, criterion, args, writer, epoch):
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
    num_images = 1

    # switch to evaluate mode
    #model.eval()
    with torch.no_grad():
        end = time.time()
        # for i, (images, target) in tqdm.tqdm(
        #     enumerate(val_loader), ascii=True, total=len(val_loader)
        # ):
        # time.sleep(0.1)
        # return -1, -1, -1
        for i, (images, target) in enumerate(val_loader):
            break
            continue
            images = images.to(args.gpu)
            target = target.to(args.gpu)

            #print(images.shape, target.shape)

            # compute output
            # output = model(images)

            # loss = criterion(output, target)
            loss = torch.Tensor([0])

            # measure accuracy and record loss
            # acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
            acc1, acc5, acc10 = torch.Tensor([5]), torch.Tensor([5]), torch.Tensor([5])
            # losses.update(loss.item(), images.size(0))
            # top1.update(acc1.item(), images.size(0))
            # top5.update(acc5.item(), images.size(0))
            # top10.update(acc10.item(), images.size(0))
            # compute weighted sum for each accuracy so we can average it later
            top1 += acc1.item()# * images.size(0)
            top5 += acc5.item()# * images.size(0)
            top10 += acc10.item()# * images.size(0)
            num_images += 1#images.size(0)

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

def do_something():
    time.sleep(5)
    print("SLEPT FOR 5s")
    return 1


if __name__ == "__main__":
    main()
