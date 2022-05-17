import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args_helper import parser_args
from torch.utils.data import random_split

import pdb
import data.wiki_data as data

class Wiki:
    def __init__(self, args):
        super(Wiki, self).__init__()

        #data_root = os.path.join(parser_args.data, "cifar100")
        #use_cuda = torch.cuda.is_available()
        #kwargs = {"num_workers": parser_args.workers, "pin_memory": True} if use_cuda else {}
        device = torch.device("cuda:{}".format(parser_args.gpu))# if args.cuda else "cpu")
        
        # Data loading code
        corpus = data.Corpus('word_language_model_revised/data/wikitext-2')

        eval_batch_size = 10
        train_data = self.batchify(corpus.train, parser_args.batch_size)
        val_data = self.batchify(corpus.valid, eval_batch_size)
        test_data = self.batchify(corpus.test, eval_batch_size)

    
        pdb.set_trace()

        self.train_loader = None
        # torch.utils.data.DataLoader(
        #     train_dataset, batch_size=parser_args.batch_size, shuffle=True, **kwargs
        # )

        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=parser_args.batch_size, shuffle=False, **kwargs
        )

        self.actual_val_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=parser_args.batch_size, shuffle=True, **kwargs
        )

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)