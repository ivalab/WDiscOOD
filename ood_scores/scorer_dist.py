import os, sys
import numpy as np
import pdb

from scipy.linalg import eigh
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

if __package__ in [None, '']:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ood_scores.scorer_base import ScorerBaseFeature
from utils.utils import _to_np, _to_tensor

from tqdm import tqdm

class FeatStaticBase(ScorerBaseFeature):
    """Fit feature statistics
    Although features are stored as numpy.array for using fast libraries,
    the statistics, such as covariance mats, are stored as torch.tensor to use cuda for fast calculation.
    """
    def __init__(self, args) -> None:
        super().__init__(args=args)
        # parse the alpha for the discriminant calculation
        self.alpha = self.args.alpha if hasattr(self.args, "alpha") else 1e-3

        # device
        self.device = self.args.device

        # the info to be obtained - NOTE: put everything to torch.tensor on cuda to speed up calculation
        self.center_cls= torch.tensor([])           # The centers for each class
        self.center = torch.tensor([])              # The all-data center
        self.Nc = torch.tensor([])                  # Class population (num_class, )
        self.scatter_within = torch.tensor([])
        self.scatter_between = torch.tensor([])
        self.scatter_total = torch.tensor([])
        self.sb_eigvecs = torch.Tensor([])
        self.sb_eigvals = torch.Tensor([])

        self.covariance = torch.tensor([])
        self.scatterness = torch.tensor([])
        self.scatter_dirs = torch.tensor([])
    
        # normalizer - for those who need it
        self.normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)

    def fit(self):
        """
        Get the directions that 
        Should be called after the all the ID features are appended
        """
        # preprocess feature before fitting
        if self.args.feat_norm:
            print("Normalizing the training features...")
            self.id_feats = self.normalizer(self.id_feats)
            print("Done.")
        # number of class
        self.num_class = self.id_labels.max() + 1

        # == prepare
        assert self.data_added(), "Please populate the features first using append_features"
        self.id_feats = self.id_feats.astype(np.float32)    

        # == class population and centers
        self.center, self.center_cls, self.Nc = self._fit_mean_population(self.id_feats, self.id_labels)
        
        # == scatter matrices
        self.scatter_within, self.scatter_between, self.scatter_total = self._fit_scatter_mats(self.id_feats, self.id_labels, self.center, self.center_cls)
        self.sb_eigvals, self.sb_eigvecs = torch.linalg.eig(self.scatter_between) 
        self.sb_eigvals, self.sb_eigvecs = torch.real(self.sb_eigvals), torch.real(self.sb_eigvecs)
        sort_idx = torch.flip(torch.argsort(self.sb_eigvals), dims=(-1, ))
        self.sb_eigvals = self.sb_eigvals[sort_idx]
        self.sb_eigvecs = self.sb_eigvecs[sort_idx]
        
        # == covariance
        self.covariance = self.scatter_within
        self.precision = torch.linalg.pinv(self.covariance)

        # == fit discriminant and scatter principle directions
        self._fit_pDirs()

        # change the status, finish 
        self.has_fit = True
    
    def _fit_mean_population(self, id_feats, id_labels):
        """Processing the numpy array and return tensor"""
        num_class = id_labels.max() + 1
        N, D = id_feats.shape
        center = np.mean(id_feats, axis=0)
        center = _to_tensor(center, device=self.device)
        center_cls = torch.zeros(size=(num_class, D), device=self.device)
        Nc = torch.zeros(size=(num_class, ), dtype=int, device=self.device)
        for i in tqdm(range(num_class)):

            # fetch out the data belongs to this class
            class_idx = np.where(id_labels == i)[0]
            class_feats = id_feats[class_idx, :]

            # class population
            Nc[i] = class_feats.shape[0]

            # class center
            center_cls[i] = _to_tensor(np.mean(class_feats, axis=0), device=self.device)

        return center, center_cls, Nc
    
    def _fit_scatter_mats(self, id_feats, id_labels, center, center_cls):
        scatter_b_es = EmpiricalCovariance(assume_centered=True)
        scatter_w_es = EmpiricalCovariance(assume_centered=True)

        # between-class
        print("Fitting the between-class scatter matrix")
        centers_centered = center_cls - center.view((1, -1))
        scatter_b_es.fit(_to_np(centers_centered))
        scatter_between = _to_tensor(scatter_b_es.covariance_, device=self.device)

        # the within-class scatter matrix
        print("Fitting the within-class scatter matrix")
        center_samples = _to_np(center_cls)[id_labels, :]
        train_feats_centered = id_feats - center_samples
        scatter_w_es.fit(_to_np(train_feats_centered))
        scatter_within = _to_tensor(scatter_w_es.covariance_, device=self.device)
           
        # the total scatter matrix - NOTE: we can do this since we sampled balanced number per class.
        scatter_total = scatter_within + scatter_between

        return scatter_within, scatter_between, scatter_total


    def _fit_pDirs(self):
        """Fit principle directions and their scatterness"""
        self.scatterness, self.scatter_dirs = torch.linalg.eig(
            self.covariance
        )
        self.scatterness = torch.real(self.scatterness)
        self.scatter_dirs = torch.real(self.scatter_dirs)

        # sort in descend order according to scatterness
        sort_idx = torch.argsort(self.scatterness)
        sort_idx = torch.flip(sort_idx, dims=(0, ))
        self.scatterness = self.scatterness[sort_idx]
        self.scatter_dirs = self.scatter_dirs[:, sort_idx]

        self.scatter_dirs = self.scatter_dirs.T

    
    def plot_stats(self):
        # feature norms histogram
        plt.figure()
        ax = plt.gca()
        feat_norm = torch.linalg.norm(self.id_feats, axis=1)
        sns.distplot(_to_np(feat_norm), hist=True, ax=ax, color='b', label="Feature norm distribution", bins=100, kde_kws={})

        # scatterness histogram
        plt.figure()
        ax = plt.gca()
        sns.distplot(_to_np(self.scatterness), hist=True, ax=ax, color='b', label="Scatterness histogram", bins=20, kde_kws={})
    
    def clear_cache(self):
        super().clear_cache()
        self.center_cls= torch.tensor([])           # The centers for each class
        self.center = torch.tensor([])              # The all-data center
        self.Nc = torch.tensor([])                  # Class population (num_class, )
        self.scatter_within = torch.tensor([])
        self.scatter_between = torch.tensor([])
        self.scatter_total = torch.tensor([])
        self.sb_eigvecs = torch.Tensor([])
        self.sb_eigvals = torch.Tensor([])

        self.covariance = torch.tensor([])
        self.scatterness = torch.tensor([])
        self.scatter_dirs = torch.tensor([])


class Maha(FeatStaticBase):
    """Mahalanobis distance w.r.t class center
    """
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def fit(self):
        super().fit()

    def cal_score(self, feat):
        """Negative Mahalanobis scores.

        Args:
            feats (D, ):          The test features
        Return:
            score (1, ):          The IDness of the sample
        """
        if self.args.feat_norm:
            feat = self.normalizer(feat)
        feat = _to_tensor(feat, device=self.device)

        feat_c = feat[None, :] - self.center_cls[:, :]
        maha_dists = torch.diag(feat_c @ self.precision @ feat_c.T)
        score = - maha_dists.min()
        return _to_np(score)

    
    def cal_score_dirs(self, feat):
        """
        Calculate the OOD scores along each scatter directions.
        It is defined as the minimum euclidean distance to the class center over all classes.

        Args:
            feats (D)             query features. 
        
        Returns:
            scores (D)          The OOD scores along D discriminants in a descending order 
                                according to the fisher discriminant ratio
        """
        feat = _to_tensor(feat, device=self.device)
        scores = torch.zeros_like(feat)

        # scores
        diff_all = feat[None, :] - self.center_cls     # (N_cls, D)
        diff_all_proj = diff_all @ (self.scatter_dirs.T)   # (N_cls, D)

        scores = - torch.pow(torch.min(diff_all_proj, dim=0)[0], 2)

        return _to_np(scores)
    
    def debug(self):
        # check the non-zero scatterness
        th = self.args.proj_error_th
        above_th_flags = self.scatterness > th
        print(f"Number of fisher ratio above threshold {th}: {torch.count_nonzero(above_th_flags)}")

        # check the dimension of scatter directions 
        print(f"THe rank of scatter directions: {torch.linalg.matrix_rank(self.scatter_dirs)}")
        print(f"THe rank of scatter dirs with scatteredness above th {th}: {torch.linalg.matrix_rank(self.scatter_dirs[above_th_flags, :])}")



class DiscBase(FeatStaticBase):
    """Discrimination Distance for the OOD detection
    """
    def __init__(self, args):
        super().__init__(args)
        self.discriminants = torch.tensor([]) # (N=D, D)
        self.fisher_ratio = torch.tensor([]) # (N=D, D)
    
    
    def fit(self):
        super().fit()
        self.discriminants, self.fisher_ratio = self._fit_disc(self.scatter_within, self.scatter_between) 

        keep_idx = self.fisher_ratio > self.args.proj_error_th
        discriminants = self.discriminants[keep_idx, :]
        fisher_ratio = self.fisher_ratio[keep_idx]
        self.disc_scale_mat = discriminants.T @ torch.diag(fisher_ratio) @ discriminants
    
    def clear_cache(self):
        super().clear_cache()
        self.discriminants = torch.tensor([])
        self.fisher_ratio = torch.tensor([])

    def _fit_disc(self, scatter_within, scatter_between):
        discriminants, fisher_ratio = self._fit_disc_naive(scatter_within, scatter_between)
        return discriminants, fisher_ratio

    def _fit_disc_naive(self, scatter_within, scatter_between):
        D = scatter_within.shape[0]
        # the discriminant directions. Currently in ascend order
        fisher_ratio, discriminants = torch.linalg.eig(
            torch.linalg.inv(scatter_within + self.alpha * torch.eye(D, device=self.device)) @ scatter_between,
        )

        ## keep real
        # NOTE: It is supposed to be real according to simultaneous diagonalization theorem, but complex exists with negligible imaginary parts due to numerical reason
        fisher_ratio, discriminants = torch.real(fisher_ratio), torch.real(discriminants)
        discriminants = discriminants.T   # To (N, D)

        # Sort in descend order w.r.t the fisher discriminant ratio
        sort_idx = torch.argsort(fisher_ratio)
        sort_idx = torch.flip(sort_idx, dims=(0, ))
        fisher_ratio = fisher_ratio[sort_idx]
        discriminants = discriminants[sort_idx, :]

        return  discriminants, fisher_ratio
        

class WDiscOOD(DiscBase):
    def __init__(self, args):
        super().__init__(args)

        self.discriminants = torch.tensor([]) # (N=D, D)
        self.fisher_ratio = torch.tensor([]) # (N=D, D)
        self.white_proj_mat = torch.tensor([])
        self.g_proj_mat = torch.tensor([])
        self.h_proj_mat = torch.tensor([])

        self.center_cls_disc = torch.tensor([])
        self.center_cls_remain = torch.tensor([])
        self.center_disc = torch.tensor([])
        self.center_remain = torch.tensor([])
    
    
    def fit(self):
        # preprocess feature before fitting
        if self.args.feat_norm:
            print("Normalizing the training features...")
            id_feats = self.normalizer(self.id_feats)
            print("Done.")
        else:
            id_feats = self.id_feats

        if self.args.whiten_data:
            feats_processed = self.preprocess_train_features(id_feats)
        else:
            feats_processed = id_feats

        # == class population and centers
        print("Fitting the class mean & population information")
        self.center, self.center_cls, self.Nc = self._fit_mean_population(feats_processed, self.id_labels)
        
        # == scatter matrices
        print("Fitting the scatter matrices")
        self.scatter_within, self.scatter_between, self.scatter_total = self._fit_scatter_mats(feats_processed, self.id_labels, self.center, self.center_cls)
        
        # == covariance
        self.covariance = self.scatter_within
        self.precision = torch.linalg.pinv(self.covariance)

        # == discriminants
        self.discriminants, self.fisher_ratio = self._fit_disc(self.scatter_within, self.scatter_between) 

        # == Now create disentangle projection
        # discriminant projection matrix.
        if self.args.num_disc is not None:
            print(f"[Discriminative space projection] Using the discriminant number: {self.args.num_disc}")
            discriminants = self.discriminants[:self.args.num_disc, :]
        else:
            print(f"[Discriminative Space projection] Using all discriminants with positive fisher ratio.")
            discriminants = self.discriminants[self.fisher_ratio>0, :]
        self.g_proj_mat = _to_tensor(discriminants, device=self.device)

        # null space projection matrix.
        if self.args.num_disc_res is None:
            print("[Residual space projection] Using the same set of discriminants as Discriminative space")
            discriminants_res = discriminants
        else:
            print(f"[Residual space projection] Using the discriminant number: {self.args.num_disc_res}")
            discriminants_res = self.discriminants[:self.args.num_disc_res, :]

        # SVD on discriminants
        _, S, basis = torch.linalg.svd(discriminants_res)
        basis = basis[:S.shape[0], :]
        S, basis = torch.real(S), torch.real(basis)
        self.h_proj_mat = torch.eye(self.D, device=self.device) - basis.T @ basis


        # The precision matrix in each space
        self.precision_cls_disc = torch.linalg.pinv(self.g_proj_mat @ self.g_proj_mat.T)
        self.precision_cls_remain = torch.linalg.pinv(self.h_proj_mat @ self.h_proj_mat.T)
        self.precision_center_disc = torch.linalg.pinv(self.g_proj_mat @ self.scatter_total @ self.g_proj_mat.T)
        self.precision_center_remain = torch.linalg.pinv(self.h_proj_mat @ self.scatter_total @ self.h_proj_mat.T)

        # project centers
        self.center_cls_disc = self.center_cls @ (self.g_proj_mat.T)
        self.center_cls_remain = self.center_cls @ (self.h_proj_mat.T)
        self.center_disc = self.g_proj_mat @ self.center
        self.center_remain = self.h_proj_mat @ self.center

        # == Get distance weight
        self.res_dist_weight = self.args.res_dist_weight
    
    def preprocess_train_features(self, feats):
        """Fit whitening matrix from training data & whiten the training features
        """
        print("Fiting class centers for getting Whitening projection matrix...")
        _, center_cls, _ = self._fit_mean_population(feats, self.id_labels)
        ec = EmpiricalCovariance(assume_centered=True)
        center_samples = _to_np(center_cls)[self.id_labels, :]
        train_feats_centered = feats - center_samples
        ec.fit(train_feats_centered)
        precision = ec.precision_

        # map them 
        eigvals, eigvecs = np.linalg.eig(precision)
        self.white_proj_mat = eigvecs @ np.diag(np.power(eigvals, 0.5)) @ eigvecs.T
        self.white_proj_mat = _to_tensor(self.white_proj_mat, device=self.device)
        feats_scaled = np.zeros_like(feats)
        print("Whitening the features...")
        for i in tqdm(range(self.N)):
            feats_scaled[i, :] = _to_np(self.white_proj_mat @ _to_tensor(feats[i, :], device=self.device))
        
        return feats_scaled

    
    def decompose_feature(self, feat):
        """Return the discriminant component and orthogonal component of feature"""
        # first standardize
        if self.args.whiten_data:
            feat_p = self.white_proj_mat @ feat
        else:
            feat_p = feat
        # then disentangle
        feat_disc = self.g_proj_mat @ feat_p
        feat_residual = self.h_proj_mat @ feat_p
        return feat_disc, feat_residual

    
    def cal_score(self, feat, return_all=False, res_dist_weight=None):
        """Calculate the OOD score given a feature vector

        Args:
            feat (array (D, )):         The query feature vector

        Returns:
            score (float):              The OOD score. Lower for OOD
        """
        if self.args.feat_norm:
            feat = self.normalizer(feat)

        # decompose feature
        feat = _to_tensor(feat, device=self.device)    
        feat_disc, feat_residual = self.decompose_feature(feat)

        # first the discrimination space score - Just euclidean for now
        dist_disc = self._cal_dist(feat_disc, option=self.args.score_g, \
            cls_centers=self.center_cls_disc, center=self.center_disc,
            cls_precision=self.precision_cls_disc,
            center_precision = self.precision_center_disc
            )
        score_disc = - dist_disc

        # the residual space score - Use shared center for now
        dist_res = self._cal_dist(feat_residual, option=self.args.score_h, \
            cls_centers=self.center_cls_remain, center=self.center_remain,
            cls_precision=self.precision_cls_remain,
            center_precision=self.precision_center_remain
            )
        score_residual = - dist_res

        if res_dist_weight is None:
            res_dist_weight = self.res_dist_weight
        score = score_disc + res_dist_weight * score_residual

        if return_all:
            return _to_np(score), _to_np(score_disc), _to_np(score_residual)
        else:
            return _to_np(score)
    
    def _cal_dist(self, feat, option="ClsEucl", cls_centers=None, center=None, cls_precision=None, center_precision=None):
        if option == "CenterEucl":
            dist = torch.norm(feat - center, p=2)
            return dist
        elif option == "ClsEucl":
            dists = torch.norm(feat[None, :] - cls_centers, p=2, dim=1)
            return torch.min(dists)
        elif option == "CenterMaha":
            diff = feat - center
            maha_dist = (diff @ center_precision @ diff)**0.5
            return maha_dist
        elif option == "CenterClsMaha":
            diff = feat - center
            maha_dist = (diff @ cls_precision @ diff)**0.5
            return maha_dist
        elif option == "ClsMaha":
            diff = feat[None, :] - cls_centers
            maha_dists = torch.diag(diff @ cls_precision @ diff.T)**0.5
            return torch.min(maha_dists)
        elif option == "Norm":
            dist = torch.norm(feat, p=2)
            return dist
        else:
            raise NotImplementedError
