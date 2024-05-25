# trying to test prospr using their codebase

import importlib
import re

from train import *
import argparse
import random
from pathlib import Path

import utils
from cli import parse_args
import prospr.utils
import random
import numpy as np
import os
import copy

from models.resnet20 import _weights_init

WEIGHT_REINIT_SANITY = False
MASK_SHUFFLE_SANITY = True

args = parse_args()
hparams = utils.Hyperparameters(**vars(args))

log_dir = utils.create_logdir("sreeniva_debug_prospr")

utils.set_seed(hparams.seed, hparams.allow_nondeterminism)

train_data, _, test_data, _ = dataloader_factory(
    hparams.dataset, hparams.batch_size
)

model = model_factory(hparams.model, hparams.dataset, hparams.no_model_patching)
model, masks = get_pruned_model(model, hparams)

model_ckpt = torch.load("/workspace/pruning_is_enough/prospr_ckpts/sp95/trained_model.pt")
mask_ckpt = torch.load("/workspace/pruning_is_enough/prospr_ckpts/sp95/pruning_keep_mask.pt")
model.load_state_dict(model_ckpt)
masks = mask_ckpt

filter_fn = pruning_filter_factory(10, hparams.structured_pruning)
structured = False

pruned_model = prospr.utils.apply_masks_with_hooks(model, masks, structured, filter_fn)

# from now onwards, only refer to pruned_model!!!
optimizer, lr_scheduler = get_optimizer(pruned_model, hparams)
pruned_model = pruned_model.cuda()

test_loss, test_acc1, test_acc5 = evaluate(pruned_model, test_data)
print("Test Accuracy: {}%".format(test_acc1*100))
orig_test_acc = test_acc1*100

weight_only = []
for name, param in pruned_model.named_parameters():
    if "weight" in name and "bn" not in name:
        weight_only.append((name, param))

# manually apply mask to model as sanity check
idx = 0
for name, param in pruned_model.named_parameters():
    if "weight" in name and "bn" not in name:
        param.data *= masks[idx]
        idx += 1

# this should be the same as before
test_loss, test_acc1, test_acc5 = evaluate(pruned_model, test_data)
print("Test Accuracy: {}%".format(test_acc1*100))


# Run sanity checks
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # making sure GPU runs are deterministic even if they are slower
    torch.backends.cudnn.deterministic = True
    # this causes the code to vary across runs. I don't want that for now.
    # torch.backends.cudnn.benchmark = True
    print("Seeded everything: {}".format(seed))

set_seed(96)

bkp_pruned_model = copy.deepcopy(pruned_model)

# weight reinit sanity check
if WEIGHT_REINIT_SANITY:
    pruned_model = pruned_model.apply(_weights_init)
    print("Running Weight Reinit Sanity Check")
else:
    shuffled_masks = []
    for mask in masks:
        idx = torch.randperm(mask.nelement())
        shuffled_masks.append(mask.view(-1)[idx].view(mask.size()))
    print("Running Mask Shuffle Sanity Check")
    masks = shuffled_masks

pruned_model = prospr.utils.apply_masks_with_hooks(pruned_model, masks, structured, filter_fn)
# manually apply mask to model as sanity check
idx = 0
for name, param in pruned_model.named_parameters():
    if "weight" in name and "bn" not in name:
        print("Pruning Layer: {}".format(name))
        print("Before: {}".format(torch.norm(param)))
        param.data *= masks[idx]
        print("After: {}".format(torch.norm(param)))
        idx += 1

# compare norms to verify that reinit has really happened.
print("Conv1.weight norm: Before={} | After={}".format(torch.norm(bkp_pruned_model.conv1.weight), torch.norm(pruned_model.conv1.weight)))

# this should be the same as before
test_loss, test_acc1, test_acc5 = evaluate(pruned_model, test_data)
print("Test Accuracy after flipping things (should be terrible): {}%".format(test_acc1*100))

# train for 200 epochs and see what happens
optimizer, lr_scheduler = get_optimizer(pruned_model, hparams)
for epoch in range(1, 200 + 1):
    avg_train_loss, epoch_time = train_one_epoch(pruned_model, train_data, optimizer)
    test_loss, test_acc1, test_acc5 = evaluate(pruned_model, test_data)

    print(
        f"📸 Epoch {epoch} (finished in {epoch_time})\n",
        f"\tTrain loss:\t{avg_train_loss:.4f}\n",
        f"\tTest loss:\t{test_loss:.4f}\n",
        f"\tTest acc:\t{test_acc1:.4f}\n",
        f"\tTest top-5 acc:\t{test_acc5:.4f}",
    )

    lr_scheduler.step()

print(
    "✅ Training finished\n",
    f"\tFinal test acc: {test_acc1}\n",
    f"\tFinal test acc@5: {test_acc5}",
)

# sanity check by applying the mask and testing accuracy
# manually apply mask to model as sanity check
idx = 0
for name, param in pruned_model.named_parameters():
    if "weight" in name and "bn" not in name:
        print("Pruning Layer: {}".format(name))
        print("Before: {}".format(torch.norm(param)))
        param.data *= masks[idx]
        print("After: {}".format(torch.norm(param)))
        idx += 1

# this should be the same as before
test_loss, test_acc1, test_acc5 = evaluate(pruned_model, test_data)
print("Test Accuracy: {}%".format(test_acc1*100))
final_test_acc = test_acc1*100

print("Sanity check complete!")
print("Accuracy before reinit: {} | Accuracy after weight reinit: {}".format(orig_test_acc, final_test_acc))

