import numpy as np
import torch
from utils.utils import _to_np, _to_tensor

class ScorerBase():
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
    def fit(self):
        """
        Calibrate the scorer
        """
        print("No fitting is needed.")
        return

    def cal_score(self):
        """
        API for calculating the OOD scores. 
        Let the ID data has higher scores
        """
        raise NotImplementedError
        
    def debug(self):
        print("No debug is needed.")
    
class ScorerBaseFeature(ScorerBase):
    """The base OOD scorer in the feature space
    """
    def __init__(self, args):
        super().__init__(args)

        self.N = -1
        self.D = -1
        self.num_class = -1
        self.id_feats = np.array([])        #(N, D), the in-distribution training features
        self.id_labels = np.array([], dtype=int)       #(N), the in-distribution data label
    
    def append_features(self, feats_new, labels_new=None, ID = True):
        """
        append the ID features

        @param[in] feats_new         np.ndarray. A new batch of features. (N, D)
        @param[in] labels_new        np.ndarray. A new batch of GT label ids. (N, )
        """
        feats_new = _to_np(feats_new)
        labels_new = _to_np(labels_new).astype(int)

        if ID: 
            if self.id_feats.size == 0:
                self.id_feats = feats_new
                self.id_labels = labels_new
            else:
                self.id_feats = np.concatenate(
                    (self.id_feats, feats_new),
                    axis = 0
                )
                self.id_labels = np.concatenate(
                    (self.id_labels, labels_new),
                    axis = 0
                )
            self.N, self.D = self.id_feats.shape
            self.num_class = self.id_labels.max() + 1

    def data_added(self):
        return self.N != -1
    
    def clear_cache(self):
        """Clear the stored id features"""
        self.N, self.D, self.num_class = -1, -1, -1
        self.id_feats = np.array([])
        self.id_labels = np.array([])

    