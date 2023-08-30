"""
The Mahalanobis OOD scoring method: https://arxiv.org/abs/1807.03888
The impelementation is adopted from: https://github.com/deeplearning-wisc/large_scale_ood
"""

import os
import pdb
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegressionCV

from .scorer_base import ScorerBase
from utils.utils import _to_np, _to_tensor
from utils.mahalanobis_lib import get_Mahalanobis_score

class Mahalanobis(ScorerBase):
    def __init__(self, args):
        super().__init__(args)

        
        self.num_classes = args.num_classes

        # parse parameter storage path
        self.save_dir = os.path.join(args.save_folder, args.id_dset)
        self.save_dir = os.path.join(self.save_dir, f"{args.arch}")
        self.params_path = os.path.join(self.save_dir, 'maha_tune_results.npy')

        # parameters to be loaded
        self.model = None
        self.num_output = None
        self.sample_mean = None
        self.precision = None
        self.magnitude = None
        self.regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                    [0, 0, 1, 1])
    
    def fit(self, model):
        # model and num_output
        self.model = model
        temp_x = torch.rand(2, 3, 224, 224)
        temp_x = Variable(temp_x).cuda()
        temp_list = self.model(x=temp_x, layer_index='all')[1]
        self.num_output = len(temp_list)

        # load parameters 
        # import pdb; pdb.set_trace()
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(self.params_path, allow_pickle=True)
        sample_mean = [_to_tensor(s, device=self.args.device) for s in sample_mean]
        precision = [_to_tensor(p, device=self.args.device) for p in precision]

        self.sample_mean = sample_mean
        self.precision = precision
        self.magnitude = magnitude
        self.regressor.coef_ = lr_weights
        self.regressor.intercept_ = lr_bias
    
    def cal_score(self, x):
        """NOTE: Here x should be the input data, rather than the feature"""
        if len(x.shape) == 3:
            x = x[None, :, :, :]
        Mahalanobis_scores = get_Mahalanobis_score(x, self.model, self.num_classes, self.sample_mean, self.precision, self.num_output, self.magnitude)
        scores = -self.regressor.predict_proba(Mahalanobis_scores)[:, 1]
        return scores

