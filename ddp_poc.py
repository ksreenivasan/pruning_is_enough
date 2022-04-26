import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import re

from torch.nn.parallel import DistributedDataParallel as DDP
from ddp_args_helper import parser_args
from ddp_utils import do_something_outside
import copy

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(3072, 100)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(100, 10)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device).reshape(-1, 32*32*3), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    return accuracy

def get_model_norm(model):
    tot_norm = 0
    for name, params in model.named_parameters():
        tot_norm += torch.norm(params.data)
    return tot_norm


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    print("Parser args: gpu={}, name={}".format(parser_args.gpu, parser_args.name))
    print("Setting gpu now, let's see what happens")
    parser_args.gpu = rank
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    # model = torchvision.models.resnet18(pretrained=False).to(rank)
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform) 
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=512, sampler=train_sampler, num_workers=8)

    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=512, shuffle=False, num_workers=8)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:{}".format(rank))

    for epoch in range(5):
        print("Local Rank: {}, Epoch: {}, Training ...".format(rank, epoch))
        print("Local Rank: {} | Parser args: gpu={}, Name={}".format(rank, parser_args.gpu, parser_args.name))
        if epoch % 3 == 0:
            # prune model
            print("Rank: {} | Gonna try to prune model".format(rank))
            for name, params in ddp_model.named_parameters():
                # basically, prune everything
                if re.match('.*\.weight', name) or re.match('.*\.bias', name):
                    params.data = torch.zeros_like(params.data)

        print("Rank: {} | Model Norm: {}".format(rank, get_model_norm(ddp_model)))
        # Save and evaluate model routinely
        if epoch % 2 == 0:
            if rank == 0:
                accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                # torch.save(ddp_model.state_dict(), model_filepath)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        ddp_model.train()
        total_data_size = [0, 0]

        for data in train_loader:
            print("Rank: {} | Model Norm: {}".format(rank, get_model_norm(ddp_model)))
            inputs, labels = data[0].to(device).reshape(-1, 32*32*3), data[1].to(device)
            # print("Device: {} | Batch Size: {} | Label sum: {}".format(rank, data[1].shape[0], torch.sum(data[1])))
            total_data_size[rank] += data[1].shape[0]
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print("End of epoch total batch sizes: {}".format(total_data_size))

        do_something_outside(rank)

    print("Local rank: {} | Entering barrier".format(rank))
    dist.barrier()
    print("Local rank: {} | Past barrier".format(rank))
    cp_model = copy.deepcopy(model)
    print("Local rank: {} | Copied Model".format(rank))

    for data in train_loader:
        print("Rank: {} | Copied Model Norm: {}".format(rank, get_model_norm(cp_model)))
        inputs, labels = data[0].to(device).reshape(-1, 32*32*3), data[1].to(device)
        # print("Device: {} | Batch Size: {} | Label sum: {}".format(rank, data[1].shape[0], torch.sum(data[1])))
        total_data_size[rank] += data[1].shape[0]
        optimizer.zero_grad()
        outputs = cp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("End of epoch total batch sizes: {}".format(total_data_size))

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 5))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
    # run_demo(demo_checkpoint, world_size)
