import os, sys
from tqdm import tqdm
import pdb

import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise_distances_argmin_min

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ood_scores.scorer_base import ScorerBase
from utils.utils import load_features

class MSPScorer(ScorerBase):
    def __init__(self, args):
        super().__init__(args)
    
    def cal_score(self, probs):
        return np.max(probs)


class EnergyScorer(ScorerBase):
    def __init__(self, args):
        super().__init__(args)
    
    def cal_score(self, logits):
        """The energy score operates in the logit space."""
        energy = - np.log(np.sum(np.exp(logits)))
        return - energy

class MaxLogitScorer(ScorerBase):
    def __init__(self, args):
        super().__init__(args)
    
    def cal_score(self, logits):
        max_logit = np.max(logits)
        return max_logit
    

class KLMatchScorer(ScorerBase):
    def __init__(self, args):
        super().__init__(args)
        self.mean_softmax_train = None
        self.kl = lambda p, q: np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    def fit(self):

        feat_dim = 2048 if not "vit" in self.args.arch else 768    # NOTE: hardcode for now

        # NOTE: the feature does not matter, hence last_idx and featdim is some random number
        _, id_train_logit, id_train_label = load_features(self.args, id=True, split="train", last_idx=0, \
            featdim=feat_dim, projdim=self.args.num_classes, feat_space=0)
        softmax_id_train = softmax(id_train_logit, axis=-1)


        # get mean training prob
        print("Calculating the mean probability prediction for ID classes")
        mean_softmax_train = [softmax_id_train[id_train_label==i].mean(axis=0) for i in tqdm(range(1000))]
        self.mean_softmax_train = np.array(mean_softmax_train)
        return super().fit()


    
    def cal_score(self, prob):
        prob = prob[None, :]
        score = -pairwise_distances_argmin_min(prob, self.mean_softmax_train, metric=self.kl)[1]
        return score

