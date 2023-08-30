"""
The ODIN OOD detection method
The impelementation and parameters are adopted from: https://github.com/deeplearning-wisc/large_scale_ood
"""

import os
import pdb
import numpy as np
import torch
from torch.autograd import Variable

from .scorer_base import ScorerBase
from utils.utils import _to_np, _to_tensor

class ODIN(ScorerBase):
    def __init__(self, args):
        super().__init__(args)

        
        self.num_classes = args.num_classes

        # params
        self.temper = 1000
        self.epsilon = 0.0
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

        # model to be loaded
        self.model = None

    def fit(self, model):
        # model and num_output
        self.model = model

    
    def cal_score(self, x):
        """NOTE: Here x should be the input data, rather than the feature"""
        x = Variable(x.cuda(), requires_grad=True)
        outputs = self.model(x)[0]

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / self.temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = self.criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -self.epsilon, gradient)
        outputs = self.model(Variable(tempInputs))[0]
        outputs = outputs / self.temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        scores = np.max(nnOutputs, axis=1)

        return scores


