"""
The VIM score: https://arxiv.org/abs/2203.10807
Code adopted from original implementation: https://github.com/haoqiwang/vim 
"""
import os, sys
import pdb
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.covariance import EmpiricalCovariance
import faiss

from .scorer_base import ScorerBaseFeature 
from utils.utils import _to_np, _to_tensor
from scipy.special import logsumexp


class VIM(ScorerBaseFeature):
    def __init__(self, args):
        super().__init__(args)
        self.index = None
        self.normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

        # also need the logits of training data
        self.id_logits = np.array([])        #(N, C), the in-distribution training logits 

        # params to fit
        self.u = None
        self.alpha = None
    
    def append_features(self, feats_new, logits_new, labels_new=None, ID = True):
        """
        append the ID features

        @param[in] feats_new         np.ndarray. A new batch of features. (N, D)
        @param[in] logits_new        np.ndarray. A new batch of logits. (N, C)
        @param[in] labels_new        np.ndarray. A new batch of GT label ids. (N, )
        """
        feats_new = _to_np(feats_new)
        logits_new = _to_np(logits_new)
        labels_new = _to_np(labels_new).astype(int)

        if ID: 
            if self.id_feats.size == 0:
                self.id_feats = feats_new
                self.id_logits = logits_new
                self.id_labels = labels_new
            else:
                self.id_feats = np.concatenate(
                    (self.id_feats, feats_new),
                    axis = 0
                )
                self.id_logits = np.concatenate(
                    (self.id_logits, logits_new),
                    axis = 0
                )
                self.id_labels = np.concatenate(
                    (self.id_labels, labels_new),
                    axis = 0
                )
            self.N, self.D = self.id_feats.shape
            self.num_class = self.id_labels.max() + 1
    
    def fit(self, w, b):
        """VIM score requires the classification layer params w and b"""
        u = -np.matmul(np.linalg.pinv(w), b)
        DIM = 1000 if self.id_feats.shape[-1] >= 2048 else 512
        print('computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(self.id_feats - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        print('computing alpha...')
        vlogit_id_train = np.linalg.norm(np.matmul(self.id_feats - u, NS), axis=-1)
        alpha = self.id_logits.max(axis=-1).mean() / vlogit_id_train.mean()
        print(f'{alpha=:.4f}')

        # store
        self.u = u
        self.alpha = alpha
        self.NS = NS
        
    def cal_score(self, feats, logits):
        squeeze_output = False 
        if len(feats.shape) == 1:
            feats = feats[None, :]
            logits = logits[None, :]
            squeeze_output = True

        vlogit_id_val = np.linalg.norm(np.matmul(feats - self.u, self.NS), axis=-1) * self.alpha
        energy_id_val = logsumexp(logits, axis=-1)
        scores = -vlogit_id_val + energy_id_val

        if squeeze_output:
            return scores[0]
        else:
            return scores




class Residual(VIM):
    def cal_score(self, feats, logits):

        squeeze_output = False 
        if len(feats.shape) == 1:
            feats = feats[None, :]
            logits = logits[None, :]
            squeeze_output = True

        scores = -np.linalg.norm(np.matmul(feats - self.u, self.NS), axis=-1)

        if squeeze_output:
            return scores[0]
        else:
            return scores



