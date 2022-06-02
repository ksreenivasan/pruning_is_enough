import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args_helper import parser_args
from torch.utils.data import random_split

import pdb
import data.wiki_data as wiki_data

class Wiki:
    def __init__(self, args):
        super(Wiki, self).__init__()

        parser_args.bptt = 35
        #data_root = os.path.join(parser_args.data, "cifar100")
        #use_cuda = torch.cuda.is_available()
        #kwargs = {"num_workers": parser_args.workers, "pin_memory": True} if use_cuda else {}
        self.device = torch.device("cuda:{}".format(parser_args.gpu))# if args.cuda else "cpu")
        
        # Data loading code
        corpus = wiki_data.Corpus('word_language_model_revised/data/wikitext-2')

        eval_batch_size = 10
        train_data = self.batchify(corpus.train, parser_args.batch_size)
        val_data = self.batchify(corpus.valid, eval_batch_size)
        test_data = self.batchify(corpus.test, eval_batch_size)


        self.train_loader = []
        for batch, i in enumerate(range(0, train_data.size(0) - 1, parser_args.bptt)): #args.bptt
            data, targets = self.get_batch(train_data, i)
            #import pdb; pdb.set_trace()
            self.train_loader.append((data, targets))

        self.val_loader = []
        for batch, i in enumerate(range(0, test_data.size(0) - 1, parser_args.bptt)): #args.bptt
            data, targets = self.get_batch(test_data, i)
            self.val_loader.append((data, targets))

        self.actual_val_loader = []
        for batch, i in enumerate(range(0, val_data.size(0) - 1, parser_args.bptt)): #args.bptt
            data, targets = self.get_batch(val_data, i)
            self.actual_val_loader.append((data, targets))
    


    def get_batch(self, source, i):
        seq_len = min(parser_args.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self.device)

# class MyLoader:
#     def __init__(self, batch_size):
        
#     # def __len__(self):
#     #     return len(self._dataloader)        

#     def __iter__(self):
#         return len(self._dataloader)


# class SplitBatchLoader:
#     """
#     Utility that transforms a DataLoader that is an iterable over (x, y) tuples
#     into an iterable over Batch() tuples, where its contents are already moved
#     to the selected device.
#     """

#     def __init__(self, dataloader, device, rank, batch_size, model, hidden_container):
#         self._dataloader = dataloader
#         self._device = device
#         self._rank = rank
#         self._batch_size = batch_size
#         self._model = model
#         self._hidden_container = hidden_container

#     def __len__(self):
#         return len(self._dataloader)

#     def __iter__(self):
#         for i, batch in enumerate(self._dataloader):
#             # if i == 0:
#             #     print("Data signature", batch.text.view(-1)[0:5].numpy())
#             x = batch.text[:, self._rank * self._batch_size : (self._rank + 1) * self._batch_size]
#             y = batch.target[:, self._rank * self._batch_size : (self._rank + 1) * self._batch_size]
#             hidden = self._model.repackage_hidden(self._hidden_container["hidden"])
#             yield Batch(x, y, hidden)


# class Batch:
#     def __init__(self, x, y, hidden):
#         self.x = x
#         self.y = y
#         self.hidden = hidden


# def train_iterator(self, batch_size: int) -> Iterable[Batch]:
#     """Create a dataloader serving `Batch`es from the training dataset.
#     Example:
#         >>> for batch in task.train_iterator(batch_size=64):
#         ...     batch_loss, gradients = task.batchLossAndGradient(batch)
#     """
#     self._epoch += 1
#     rank = torch.distributed.get_rank() if torch.distributed.is_available() else 1
#     self._hidden_container["hidden"] = self._model.init_hidden(batch_size)
#     return SplitBatchLoader(
#         self.train_loader,
#         self._device,
#         rank,
#         batch_size,
#         model=self._model,
#         hidden_container=self._hidden_container,
#     )