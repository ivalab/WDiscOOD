import numpy as np

import os
import sys
from tqdm import tqdm
import pdb
import torch
import torch.nn.functional as F

rDir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(rDir)
from ood_scores.scorer_base import ScorerBase
from ood_scores.scorer_maha import Mahalanobis
from utils.metrics import get_measures
from utils.utils import _to_np, print_measures


PROB_METHODS = ["MSP", "KLMatch"]
LOGITS_METHODS = ["Energy", "maxLogit"]
FEAT_METHODS = ["DiscDist", "KNN"]

class Evaluator():
    """The evaluator of the OOD

    When evaluating the OOD, the evaluator will samples a fixed number of data from the data because the AUPR criteria is sensitive to the ratio between the pos&neg sample numbers
    From each data loader the sampling will happen multiple times, and the mean and the variance will be stored (unless only sample once.)

    """
    def __init__(self, args, ood_num=None):

        self.args = args
        self.ood_num = ood_num

        # in-distribution scores
        self.in_scores = None
        self.in_correct_flags = None

        # ood_scores and/or results
        self.aurocs = []
        self.fprs = []

    def eval(self, net, loader, scorer:ScorerBase, in_dist=False):
        """
        Evaluate the classification rate and OOD scores given a network and a data loader
        If it is the in_distribution loader, will store the scores.
        If it is the OOD loader, will calculate the metrics.

        Args:
            net ([type]): [description]
            data_loader ([type]): [description]
            scorer ([type]): [description]
            in_dist (bool, optional): [description]. Defaults to False.

        Returns:
            scores (array, (N_data,) ):         The OOD scores
            auroc (float):                      The auroc
            fpr (float):                        The fpr
        """
        # deal with special ones
        if self.args.score in ["Maha", "ODIN"]:
            return self.eval_maha_odin(loader, scorer, in_dist)

        # sample number - For ood, allow setting the sample number
        num_samples = len(loader.dataset)
        if not in_dist and self.ood_num is not None:
            if len(loader.dataset) > self.ood_num:
                print(f"Too many samples. Will sample {self.ood_num} out of {len(loader.dataset)}")
                num_samples = self.ood_num
        
        # ==================== get started
        scores = []
        correct_flags = []
        net.eval()
        with tqdm(total=num_samples) as pbar:
            for batch_idx, (inputs, labels) in enumerate(loader):
                #if batch_idx > 5:
                #    break
                # import pdb;pdb.set_trace()
                inputs = inputs.to(self.args.device)
                start_ind = batch_idx * self.args.test_bs 
                end_ind = min((batch_idx + 1) * self.args.test_bs, num_samples)

                # get network outputs
                logits, feats = net(inputs)
                if self.args.score in PROB_METHODS:
                    probs = F.softmax(logits, dim=1)
                    out = probs
                elif self.args.score in LOGITS_METHODS:
                    out = logits
                elif self.args.score in FEAT_METHODS:
                    out = feats
                else:
                    raise NotImplementedError

                # get correct indices
                logits = _to_np(logits)
                labels = _to_np(labels)
                preds = np.argmax(logits, axis=1)
                if in_dist:
                    correct_flags.append(preds == labels)

                # calculate scores
                out = _to_np(out)
                for i in range(out.shape[0]):
                    scores_this = scorer.cal_score(out[i])
                    scores.append(scores_this)

                # update bar; quit if sampled enough data
                pbar.update(self.args.test_bs)
                if end_ind >= num_samples:
                    break
        scores = np.array(scores)
        # ==================== 

        # if ID, get the classification error rate, and save the results
        if in_dist:
            print("The in-distribution data: {}".format(loader.dataset.name))
            self.in_scores = scores
            self.in_correct_flags = np.concatenate(correct_flags, axis=0)
            cls_acc = np.count_nonzero(self.in_correct_flags) / self.in_correct_flags.size
            print(f'In-distribution Classification Top-1 Accuracy: {cls_acc} \n\n')
            # import pdb; pdb.set_trace()
        # If OOD, get the evaluation result
        else:
            # save
            auroc_this, _, fpr_this = get_measures(self.in_scores, scores)
            self.aurocs.append(auroc_this); self.fprs.append(fpr_this)

            # print
            print(f"{loader.dataset.name} OOD Detection Results:")
            print_measures(auroc_this, fpr_this, method_name=self.args.score)
    
    def eval_maha_odin(self, loader, scorer:Mahalanobis, in_dist=False):
        # sample number - For ood, allow setting the sample number
        num_samples = len(loader.dataset)
        if not in_dist and self.ood_num is not None:
            if len(loader.dataset) > self.ood_num:
                print(f"Too many samples. Will sample {self.ood_num} out of {len(loader.dataset)}")
                num_samples = self.ood_num
        
        # ==================== get started
        scores = []
        correct_flags = []
        with tqdm(total=num_samples) as pbar:
            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(self.args.device)
                start_ind = batch_idx * self.args.test_bs 
                end_ind = min((batch_idx + 1) * self.args.test_bs, num_samples)

                scores_this = scorer.cal_score(inputs)
                scores.append(scores_this)

                # update bar; quit if sampled enough data
                pbar.update(self.args.test_bs)
                if end_ind >= num_samples:
                    break
        scores = np.concatenate(scores)
        # ==================== 

        # if ID, get the classification error rate, and save the results
        if in_dist:
            print("The in-distribution data: {}. OOD eval complete".format(loader.dataset.name))
            self.in_scores = scores
        # If OOD, get the evaluation result
        else:
            # save
            auroc_this, _, fpr_this = get_measures(self.in_scores, scores)
            self.aurocs.append(auroc_this); self.fprs.append(fpr_this)

            # print
            print(f"{loader.dataset.name} OOD Detection Results:")
            print_measures(auroc_this, fpr_this, method_name=self.args.score)
    
    def print_mean_results(self):
        print(f"\n\nMean OOD Detection Results:")
        print_measures(np.mean(self.aurocs), np.mean(self.fprs), method_name=self.args.score)
