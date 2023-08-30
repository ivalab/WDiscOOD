import os, sys
import numpy as np
import pdb

from scipy.linalg import eigh
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
import seaborn as sns
import torch

if __package__ in [None, '']:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ood_scores.scorer_dist import FeatStaticBase , DiscDist
from utils.utils import _to_np, _to_tensor

class MapperBase():
    def __init__(self) -> None:
        pass

    def fit(self):
        return
    
    def map(self, feat):
        return feat

class SbSpaceMapper(DiscDist, MapperBase):
    def __init__(self, args, ortho=False, center_style=None) -> None:
        super().__init__(args)
        self.center_style = center_style

        self.proj_error_th = args.proj_error_th
        self.basis = torch.tensor([])
        self.proj_mat = torch.tensor([])

        self.ortho = ortho
    
    def fit(self, w=None, b=None):
        if w is not None and b is not None:
            self.u = _to_tensor(-np.matmul(np.linalg.pinv(w), b), device=self.device)

        if self.center_style is None:
            pass
        elif self.center_style == "CENTER":
            self.id_feats = self.id_feats - np.mean(self.id_feats, axis=0, keepdims=True)
        elif self.center_style == "VIM":
            self.id_feats = self.id_feats - _to_np(self.u[None, :])
        else:
            raise NotImplementedError

        # fit after centerized
        super().fit()
        # ======= Fit the mapping matrix
        # first filter the sb_eigvecs based on eigvals
        keep_flags = self.sb_eigvals > self.proj_error_th
        sb_eigvecs = self.sb_eigvecs[keep_flags, :]

        # SVD on eigenvecs
        _, S, basis = torch.linalg.svd(sb_eigvecs)
        basis = basis[:S.shape[0], :]
        S, basis = torch.real(S), torch.real(basis)
        self.basis = basis

        # the mapping function
        if not self.ortho:
            self.proj_mat = basis.T @ basis
        else:
            self.proj_mat = torch.eye(self.D, device=self.device) - basis.T @ basis
    
    def map(self, feat):
        feat = _to_tensor(feat, device=self.device)
        return _to_np(self.proj_mat @ feat)
    
    def debug(self):
        if self.ortho:
            error = self.proj_mat @ self.discriminants.T
            print(f"The proj_mat x discriminants error. Expect to be all zero, but it is not. {error}")
            error = self.proj_mat @ self.sb_eigvecs[self.sb_eigvals > self.proj_error_th, :].T
            print(f"The proj_mat x sb_eigvecs error. Expect to be all zero. {error}")
        print(f"The dimensionality of the Sb space under the th {self.proj_error_th}: {self.basis.shape[0]}")


class DiscSpaceMapper(DiscDist, MapperBase):
    def __init__(self, args, ortho=False, center_style=None) -> None:
        super().__init__(args)
        self.center_style=center_style

        self.proj_error_th = args.proj_error_th
        self.basis = torch.tensor([])
        self.proj_mat = torch.tensor([])

        self.ortho = ortho
    
    def fit(self):
        super().fit()
        # ======= Fit the mapping matrix
        # first filter the discriminants based on fisher ratio
        keep_flags = self.fisher_ratio > self.proj_error_th
        discriminants = self.discriminants[keep_flags, :]

        # SVD on eigenvecs
        _, S, basis = torch.linalg.svd(discriminants)
        basis = basis[:S.shape[0], :]
        S, basis = torch.real(S), torch.real(basis)
        self.basis = basis

        # the mapping function
        if not self.ortho:
            self.proj_mat = basis.T @ basis
        else:
            self.proj_mat = torch.eye(self.D, device=self.device) - basis.T @ basis
    
    def map(self, feat):
        feat = _to_tensor(feat, device=self.device)
        return _to_np(self.proj_mat @ feat)
    
    def debug(self):
        if self.ortho:
            error = self.proj_mat @ self.discriminants.T
            print(f"The proj_mat x discriminants error. Expect to be all zero. {error}")
            error = self.proj_mat @ self.sb_eigvecs[self.sb_eigvals > self.proj_error_th, :].T
            print(f"The proj_mat x sb_eigvecs error. Expect to be all zero, but perhaps won't happen. {error}")
        print(f"The dimensionality of the Discriminant space under the th {self.proj_error_th}: {self.basis.shape[0]}")


class PrincipleSpaceMapper(FeatStaticBase):
    def __init__(self, args, ortho=False, center_style=None) -> None:
        super().__init__(args)

        # self.N_pd = args.N_pd
        # self.basis = torch.tensor([])
        self.proj_mat = torch.tensor([])
        self.PS = torch.tensor([])
        self.NS = torch.tensor([])

        self.ortho = ortho

        self.center_style =center_style 
        self.u = torch.tensor([])
    
    def fit(self, w=None, b=None):
        super().fit()
        print('computing principal space...')
        DIM = self.args.topk_principal
        ec = EmpiricalCovariance(assume_centered=True)
        if w is not None and b is not None:
            self.u = _to_tensor(-np.matmul(np.linalg.pinv(w), b), device=self.device)
        if self.center_style == "CENTER":
            ec.fit(self.id_feats - _to_np(self.center[None, :]))
        elif self.center_style == "VIM":
            ec.fit(self.id_feats - _to_np(self.u[None, :]))
        else:
            raise NotImplementedError
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        p_idx = np.argsort(eig_vals * -1)[:DIM]
        n_idx = np.argsort(eig_vals * -1)[DIM:]
        PS = np.ascontiguousarray((eigen_vectors.T[p_idx]).T)
        NS = np.ascontiguousarray((eigen_vectors.T[n_idx]).T)
        self.PS = _to_tensor(PS, device=self.args.device)
        self.NS = _to_tensor(NS, device=self.args.device)
        if not self.ortho:
            self.proj_mat = self.PS @ self.PS.T
        else:
            self.proj_mat = self.NS @ self.NS.T

    
    def map(self, feat):
        feat = _to_tensor(feat, device=self.device)
        if self.center_style == "CENTER":
            feat_proj = self.proj_mat @ (feat - self.center) 
        elif self.center == "VIM":
            feat_proj = self.proj_mat @ (feat - self.u)
        return _to_np(feat_proj)
    
class MahaScaledMapper(FeatStaticBase):
    def __init__(self, args, center_style=None) -> None:
        super().__init__(args)
        self.center_style = center_style 
    
    def fit(self):
        super().fit()
        print("Computing the mapping matrix...")
        mat = _to_np(self.precision)
        eigvals, eigvecs = np.linalg.eig(mat)
        self.proj_mat = _to_tensor(eigvecs @ np.diag(np.power(eigvals, 0.5)) @ eigvecs.T, device=self.device)
    
    def map(self, feat):
        feat = _to_tensor(feat, device=self.device)
        if self.center_style is None:
            feat_proj = self.proj_mat @ (feat)
        elif self.center_style == "CENTER":
            feat_proj = self.proj_mat @ (feat - self.center)
        return _to_np(feat_proj)
