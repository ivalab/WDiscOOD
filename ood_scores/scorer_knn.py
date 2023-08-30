"""
The KNN-ood score: https://arxiv.org/abs/2204.06507
Code adopted from original implementation: https://github.com/deeplearning-wisc/knn-ood
"""
import os, sys
import pdb
import numpy as np
import torch
from torch.autograd import Variable
import faiss

from .scorer_base import ScorerBaseFeature 
from utils.utils import _to_np, _to_tensor


class KNN(ScorerBaseFeature):
    def __init__(self, args):
        super().__init__(args)
        self.index = None
        self.normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
        # top-k number. The K_full is the value used when the entire training set is used for fitting.
        if self.args.id_dset == "imagenet":
            self.K_full = 1000       
        elif self.args.id_dset == "cifar100":
            self.K_full = 200
        elif self.args.id_dset == "cifar10":
            self.K_full = 50
        else:
            raise NotImplementedError

        # total train data number
        self.id_train_size_total = self.args.id_train_size_total
        self.K_scale = 1
        self.K = round(self.K_full * self.K_scale)
    
    def fit(self):
        self.index = faiss.IndexFlatL2(self.id_feats.shape[1])
        self.index.add(self.normalizer(self.id_feats))
        # fit K_scale
        self.K_scale = float(self.id_feats.shape[0]) / float(self.id_train_size_total)
        self.K = round(self.K_full * self.K_scale)
    
    def cal_score(self, feat):
        squeeze_output = False 
        if len(feat.shape) == 1:
            feat = feat[None, :]
            squeeze_output = True
        feat = self.normalizer(feat)

        D, _ = self.index.search(feat, self.K)
        scores =  -D[:, -1]
        if squeeze_output:
            return scores[0]
        else:
            return scores




