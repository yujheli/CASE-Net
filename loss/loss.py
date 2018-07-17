from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    def __init__(self,
                 use_cuda=True):
        super(ClassificationLoss, self).__init__()

        self.use_cuda = use_cuda

    def forward(self, predict, gt, weight=None):
        """
            Args:
                predict: batch size x number of classes
                gt:      batch size x 1
                
                weight:  A manual rescaling weight given to each class.
                         If given, has to be a Tensor of size "number of classes"
        """

        batch = gt.size()[0]

        if self.use_cuda:
            #loss = F.cross_entropy(predict, gt.cuda(), weight=weight)
            criterion = nn.CrossEntropyLoss().cuda()
            loss = criterion(predict, gt.cuda())
        else:
            loss = F.cross_entropy(predict, gt, weight=weight)

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self,
                 margin=2):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin

    def forward(self, predict, gt):
        """
            Args:
                predict: batch size x 2048 x 1 x 1 -> batch size x 2048
                gt:      batch size x 1
        """
        predict = predict.view(predict.size()[0], -1)
        batch, dim = predict.size()

        loss = 0
        for i in range(batch):
            for j in range(i, batch):
                if gt[i] == gt[j]:
                    label = 1
                else:
                    label = 0
                dist = torch.dist(predict[i], predict[j], p=2) ** 2 / dim
                loss += label * dist + (1 - label) * F.relu(self.margin - dist)
        loss = 2 * loss / (batch * (batch - 1))

        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self,
                 dist_metric='L1',
                 use_cuda=True):
        super(ReconstructionLoss, self).__init__()
        
        self.dist_metric = dist_metric
        self.use_cuda = use_cuda

    def forward(self, reconstruct, gt):
        """
            Args:
                reconstruct: batch size x 3 x 224 x 224
                gt:          batch size x 3 x 224 x 224

            Return:
                Averaged per-pixel reconstruction loss.
        """

        if self.dist_metric == 'L1':
            p = 1
        else:
            p = 2

        b,c,h,w = gt.size()
        
        if self.use_cuda:
            loss = torch.dist(reconstruct, gt.cuda(), p=p) / (b*h*w)
        else:
            loss = torch.dist(reconstruct, gt, p=p) / (b*h*w)

        return loss
