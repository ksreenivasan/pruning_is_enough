from transformer_main_utils import *


def main():
    print(parser_args)
    set_seed(parser_args.seed * parser_args.trial_num)
    main_worker(parser_args.gpu)


def main_worker(gpu):
    if parser_args.subfolder is not None:
        if not os.path.isdir('results/'):
            os.mkdir('results/')
        result_subroot = 'results/' + parser_args.subfolder
        if not os.path.isdir(result_subroot):
            os.mkdir(result_subroot)
    else:
        print("Please specify the subfolder name for now. Thanks!")
        exit()

    parser_args.gpu = gpu
    if parser_args.gpu is not None:
        print("Use GPU: {} for training".format(parser_args.gpu))
    device = torch.device("cuda:{}".format(parser_args.gpu))

    # LOAD DATA #
    corpus = data.Corpus(parser_args.data)
    eval_batch_size = 10
    train_data = batchify(corpus.train, parser_args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    # LOAD MODEL #
    ntokens = len(corpus.dictionary)
    model = transformer_model.TransformerModel(get_builder(), ntokens, parser_args.transformer_emsize, parser_args.transformer_nhead, parser_args.transformer_nhid, parser_args.transformer_nlayers, parser_args.transformer_dropout).to(device)
    model = set_gpu(parser_args, model)
    criterion = nn.NLLLoss()

    if not parser_args.override_prune_rate:
        parser_args.prune_rate = get_prune_rate(parser_args.target_sparsity, parser_args.iter_period)
        print("Setting prune_rate to {}".format(parser_args.prune_rate))
    else:
        print("Overriding prune_rate to {}".format(parser_args.prune_rate))

    # Loop over epochs.
    lr = parser_args.lr
    optimizer = get_optimizer(parser_args, model)
    # scheduler = get_scheduler(optimizer, parser_args.lr_policy, max_epochs=parser_args.epochs)
    best_val_loss = None

    epoch_list, val_acc_list, model_sparsity_list = [], [], []
    for epoch in range(parser_args.epochs):
        epoch_list.append(epoch)
        epoch_start_time = time.time()
        train(parser_args, epoch, ntokens, train_data, model, optimizer, criterion)
        # scheduler.step()

        val_loss = evaluate(parser_args, model, ntokens, criterion, val_data)
        val_acc_list.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join("results", parser_args.subfolder, "train_model.pt"), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss

        if not parser_args.weight_training and parser_args.algo in ['hc_iter', 'global_ep_iter'] and epoch % (parser_args.iter_period) == 0 and epoch != 1:
            prune(model)
        cp_model = round_model(model, parser_args.round, noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)
        avg_sparsity = print_nonzeros(cp_model)
        print('Model avg sparsity: {}'.format(avg_sparsity))
        model_sparsity_list.append(avg_sparsity)

    # Load the best saved model.
    with open(os.path.join("results", parser_args.subfolder, "train_model.pt"), 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(parser_args, model, ntokens, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if not parser_args.skip_fine_tune:
        finetune(parser_args, ntokens, model, criterion, epoch_list, val_acc_list, model_sparsity_list)
        


if __name__ == "__main__":
    main()

