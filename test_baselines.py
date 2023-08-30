
import pdb
import numpy as np
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset


# cyy
from utils.argparser import OODArgs
from dataset.dataloaders import get_loaders_for_ood
from models.get_models import get_model
from ood_scores.get_scorers import get_scorer
from utils.evaluator import Evaluator

# arguments
argparser = OODArgs()
args = argparser.get_args()

# ========================== Prepare
# ------ data loaders
id_train_loader, id_test_loader, ood_loaders = get_loaders_for_ood(args)
ood_num = len(id_test_loader.dataset)
print("Dataloaders ready. \n")

# ------ model. Enforce strict here.
print(f"Number of ID classes: {args.num_classes}")
net = get_model(arch=args.arch, args=args, load_file=args.load_file, strict=True)
net.eval()
if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
elif args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

cudnn.benchmark = True  # fire on all cylinders

# ------ scorer
scorer = get_scorer(args.score, args)
if args.score in ["Maha", "ODIN"]:
    scorer.fit(net)
else:
    scorer.fit()

# evaluator
evaluator = Evaluator(args, ood_num=ood_num)

# =========================== Get started
# ID test set
print(f"\n\n Calculating the scores for the ID test set...") 
evaluator.eval(net, id_test_loader, scorer, in_dist=True)

# OODs
for out_loader in ood_loaders:
    print(f"\n\nEvaluating on {args.id_dset} v.s. {out_loader.dataset.name}...")
    evaluator.eval(net, out_loader, scorer, in_dist=False)

# Mean results
evaluator.print_mean_results()



