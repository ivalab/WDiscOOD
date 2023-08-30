"""

Test the OOD DiscDist2 scorer. It disentangle the feature into discriminative space and the residual space

"""
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from ood_scores.get_scorers import get_scorer
from utils.argparser import OODArgs
from dataset.dataloaders import get_loaders_for_ood
from models.get_models import get_model
from utils.metrics import get_measures
from utils.utils import print_measures, load_features, get_feat_dims


np.random.seed(10)


# ==================== Prepare
# args
argparser = OODArgs()
args = argparser.get_args()

print(args)

# scorer
scorer = get_scorer(args.score, args)

# dataloaders
id_train_loader, id_test_loader, ood_loaders = get_loaders_for_ood(args)

# feature dims
net = get_model(arch=args.arch, args=args, load_file=args.load_file, strict=True)
net.eval()
if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
elif args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True  # fire on all cylinders

embed_mode = "supcon" in args.arch or "clip" in args.arch
proj_dim, featdims = get_feat_dims(args, net, embed_mode=embed_mode)
last_start_idx = featdims[:-1].sum()    #<- For getting the last feature only

# ==================== Load id_train features and prepare the scorer
# Load id-train feature and append
id_train_feat, id_train_logit, id_train_label = load_features(args, id=True, split="train", last_idx=last_start_idx, \
    featdim=featdims[-1], projdim=proj_dim, feat_space=args.feat_space)
scorer.append_features(id_train_feat, id_train_label)
print(f"The ID data number: {scorer.N}; Feature dim: {scorer.D}; Class number: {scorer.num_class}")
scorer.fit()
print("Scorer fitting done.")

# ==================== Load id test features and cal ood score
print("Calculating the in distribution OOD scores...")
id_test_feat, id_test_logit, id_test_label = load_features(args, id=True, split="test", last_idx=last_start_idx, \
    featdim=featdims[-1], projdim=proj_dim, feat_space=args.feat_space)
if args.run_together:
    id_scores = scorer.cal_score(id_test_feat, return_all=True)
else:
    N = id_test_feat.shape[0]
    id_scores = []
    id_scores_disc = []
    id_scores_residual = []
    for i in tqdm(range(N)):
        id_score_this, id_score_disc_this, id_score_residual_this = scorer.cal_score(id_test_feat[i, :], return_all=True) 
        id_scores.append(id_score_this)
        id_scores_disc.append(id_score_disc_this)
        id_scores_residual.append(id_score_residual_this)
    id_scores = np.array(id_scores)
    id_scores_disc = np.array(id_scores_disc)
    id_scores_residual = np.array(id_scores_residual)

# ==================== Load ood test features, cal ood score, and evaluate ood_performance
aurocs, fprs = [], []
aurocs_disc, fprs_disc = [], []
aurocs_residual, fprs_residual = [], []
ood_names = [ood_loader.dataset.name for ood_loader in ood_loaders]
for ood_name in ood_names:
    print(f"\n\n{ood_name} OOD Detection")

    # load feature
    ood_feat, ood_logit = load_features(args, id=False, ood_name=ood_name, last_idx=last_start_idx, \
        featdim=featdims[-1], projdim=proj_dim, feat_space=args.feat_space)

    # calculate ood_scores
    if args.run_together:
        ood_scores = scorer.cal_score(ood_feat, return_all=True)
    else:
        N = ood_feat.shape[0]
        ood_scores = []
        ood_scores_disc = []
        ood_scores_residual = []
        for i in tqdm(range(N)):
            ood_score_this, ood_score_disc_this, ood_score_residual_this = scorer.cal_score(ood_feat[i, :], return_all=True)
            ood_scores.append(ood_score_this)
            ood_scores_disc.append(ood_score_disc_this)
            ood_scores_residual.append(ood_score_residual_this)
        ood_scores = np.array(ood_scores)
        ood_scores_disc = np.array(ood_scores_disc)
        ood_scores_residual = np.array(ood_scores_residual)

    # evaluate - Use all the OOD samples
    for id_s, ood_s, name, aurocs_bin, fprs_bin in zip(
        [id_scores_disc, id_scores_residual, id_scores],
        [ood_scores_disc, ood_scores_residual, ood_scores],
        ["DiscSpace", "ResidualSpace", "Together"],
        [aurocs_disc, aurocs_residual, aurocs],
        [fprs_disc, fprs_residual, fprs],
    ):
        auroc_this, _, fpr_this = get_measures(id_s, ood_s)
        aurocs_bin.append(auroc_this)
        fprs_bin.append(fpr_this)

        # print 
        print_measures(auroc=auroc_this, fpr=fpr_this, method_name=name)

# the mean performance
print("\n\n")
print("Mean results:")
for  name, aurocs_bin, fprs_bin in zip(
    ["DiscSpace", "ResidualSpace", "Together"],
    [aurocs_disc, aurocs_residual, aurocs],
    [fprs_disc, fprs_residual, fprs],
):
    auroc = np.array(aurocs_bin).mean()
    fpr = np.array(fprs_bin).mean()
    print_measures(auroc=auroc, fpr=fpr, method_name=name)


