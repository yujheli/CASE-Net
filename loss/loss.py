from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
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

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap, dim=0)
        dist_an = torch.stack(dist_an, dim=0)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
    
    
# """


# New added losses


# """    

# def normalize(x, axis=-1):
#     """Normalizing to unit length along the specified dimension.
#     Args:
#         x: pytorch Variable
#     Returns:
#         x: pytorch Variable, same shape as input      
#     """
#     x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
#     return x


# def euclidean_dist(x, y):
#     """
#     Args:
#         x: pytorch Variable, with shape [m, d]
#         y: pytorch Variable, with shape [n, d]
#     Returns:
#         dist: pytorch Variable, with shape [m, n]
#     """
#     m, n = x.size(0), y.size(0)
#     xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
#     yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
#     dist = xx + yy
#     dist.addmm_(1, -2, x, y.t())
#     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#     return dist


# def batch_euclidean_dist(x, y):
#     """
#     Args:
#         x: pytorch Variable, with shape [N, m, d]
#         y: pytorch Variable, with shape [N, n, d]
#     Returns:
#         dist: pytorch Variable, with shape [N, m, n]
#     """
#     assert len(x.size()) == 3
#     assert len(y.size()) == 3
#     assert x.size(0) == y.size(0)
#     assert x.size(-1) == y.size(-1)

#     N, m, d = x.size()
#     N, n, d = y.size()

#     # shape [N, m, n]
#     xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
#     yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
#     dist = xx + yy
#     dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))
#     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#     return dist


# def shortest_dist(dist_mat):
#     """Parallel version.
#     Args:
#         dist_mat: pytorch Variable, available shape:
#             1) [m, n]
#             2) [m, n, N], N is batch size
#             3) [m, n, *], * can be arbitrary additional dimensions
#     Returns:
#         dist: three cases corresponding to `dist_mat`:
#             1) scalar
#             2) pytorch Variable, with shape [N]
#             3) pytorch Variable, with shape [*]
#     """
#     m, n = dist_mat.size()[:2]
#     # Just offering some reference for accessing intermediate distance.
#     dist = [[0 for _ in range(n)] for _ in range(m)]
#     for i in range(m):
#         for j in range(n):
#             if (i == 0) and (j == 0):
#                 dist[i][j] = dist_mat[i, j]
#             elif (i == 0) and (j > 0):
#                 dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
#             elif (i > 0) and (j == 0):
#                 dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
#             else:
#                 dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
#     dist = dist[-1][-1]
#     return dist


# def local_dist(x, y):
#     """
#     Args:
#         x: pytorch Variable, with shape [M, m, d]
#         y: pytorch Variable, with shape [N, n, d]
#     Returns:
#         dist: pytorch Variable, with shape [M, N]
#     """
#     M, m, d = x.size()
#     N, n, d = y.size()
#     x = x.contiguous().view(M * m, d)
#     y = y.contiguous().view(N * n, d)
#     # shape [M * m, N * n]
#     dist_mat = euclidean_dist(x, y)
#     dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
#     # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
#     dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
#     # shape [M, N]
#     dist_mat = shortest_dist(dist_mat)
#     return dist_mat


# def batch_local_dist(x, y):
#     """
#     Args:
#         x: pytorch Variable, with shape [N, m, d]
#         y: pytorch Variable, with shape [N, n, d]
#     Returns:
#         dist: pytorch Variable, with shape [N]
#     """
#     assert len(x.size()) == 3
#     assert len(y.size()) == 3
#     assert x.size(0) == y.size(0)
#     assert x.size(-1) == y.size(-1)

#     # shape [N, m, n]
#     dist_mat = batch_euclidean_dist(x, y)
#     dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
#     # shape [N]
#     dist = shortest_dist(dist_mat.permute(1, 2, 0))
#     return dist


# def hard_example_mining(dist_mat, labels, return_inds=False):
#     """For each anchor, find the hardest positive and negative sample.
#     Args:
#         dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
#         labels: pytorch LongTensor, with shape [N]
#         return_inds: whether to return the indices. Save time if `False`(?)
#     Returns:
#         dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#         dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#         p_inds: pytorch LongTensor, with shape [N]; 
#             indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#         n_inds: pytorch LongTensor, with shape [N];
#             indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#     NOTE: Only consider the case in which all labels have same num of samples, 
#         thus we can cope with all anchors in parallel.
#     """

#     assert len(dist_mat.size()) == 2
#     assert dist_mat.size(0) == dist_mat.size(1)
#     N = dist_mat.size(0)
    
#     # shape [N, N]
#     is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
#     is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    
#     # `dist_ap` means distance(anchor, positive)
#     # both `dist_ap` and `relative_p_inds` with shape [N, 1]
   
#     # `dist_an` means distance(anchor, negative)
#     # both `dist_an` and `relative_n_inds` with shape [N, 1]
#     dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    
#     dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    
#     dist_ap = dist_ap.squeeze(1)
#     dist_an = dist_an.squeeze(1)

#     if return_inds:
#         # shape [N, N]
#         ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze( 0).expand(N, N))

#         # shape [N, 1]
#         p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
#         n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)

#         # shape [N]
#         p_inds = p_inds.squeeze(1)
#         n_inds = n_inds.squeeze(1)
#         return dist_ap, dist_an, p_inds, n_inds

#     return dist_ap, dist_an


# def GlobalLoss(tri_loss, global_feat, labels, normalize_feature=True):
#     """
#     Args:
#         tri_loss: a `TripletLoss` object
#         global_feat: pytorch Variable, shape [N, C]
#         labels: pytorch LongTensor, with shape [N]
#         normalize_feature: whether to normalize feature to unit length along the 
#             Channel dimension
#     Returns:
#         loss: pytorch Variable, with shape [1]
#         p_inds: pytorch LongTensor, with shape [N]; 
#             indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#         n_inds: pytorch LongTensor, with shape [N];
#             indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#         =============
#         For Debugging
#         =============
#         dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#         dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#         ===================
#         For Mutual Learning
#         ===================
#         dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
#     """
#     if normalize_feature:
#         global_feat = normalize(global_feat, axis=-1)
#     # shape [N, N]
#     dist_mat = euclidean_dist(global_feat, global_feat)
#     dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
#         dist_mat, labels, return_inds=True)
#     loss = tri_loss(dist_ap, dist_an)
#     #return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat
#     return loss, p_inds, n_inds


# def LocalLoss(tri_loss, local_feat, p_inds=None, n_inds=None, labels=None, normalize_feature=True):
#     """
#     Args:
#         tri_loss: a `TripletLoss` object
#         local_feat: pytorch Variable, shape [N, H, c] (NOTE THE SHAPE!)
#         p_inds: pytorch LongTensor, with shape [N]; 
#             indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#         n_inds: pytorch LongTensor, with shape [N];
#             indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#         labels: pytorch LongTensor, with shape [N]
#         normalize_feature: whether to normalize feature to unit length along the 
#             Channel dimension
    
#     If hard samples are specified by `p_inds` and `n_inds`, then `labels` is not 
#     used. Otherwise, local distance finds its own hard samples independent of 
#     global distance.
    
#     Returns:
#         loss: pytorch Variable,with shape [1]
#         =============
#         For Debugging
#         =============
#         dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#         dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#         ===================
#         For Mutual Learning
#         ===================
#         dist_mat: pytorch Variable, pairwise local distance; shape [N, N]
#     """
#     if normalize_feature:
#         local_feat = normalize(local_feat, axis=-1)
#     if p_inds is None or n_inds is None:
#         dist_mat = local_dist(local_feat, local_feat)
#         dist_ap, dist_an = hard_example_mining(dist_mat, labels, return_inds=False)
#         loss = tri_loss(dist_ap, dist_an)
#         return loss, dist_ap, dist_an, dist_mat
#     else:
#         dist_ap = batch_local_dist(local_feat, local_feat[p_inds])
#         dist_an = batch_local_dist(local_feat, local_feat[n_inds])
#         loss = tri_loss(dist_ap, dist_an)
#         #return loss, dist_ap, dist_an
#         return loss
    
    
# class TripletLoss(object):
#     """
#         Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
#         Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
#         Loss for Person Re-Identification'.
#     """
#     def __init__(self, margin=None):
#         self.margin = margin
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()

#     def __call__(self, dist_ap, dist_an):
#         """
#             Args:
#                 dist_ap: pytorch Variable, distance between anchor and positive sample, 
#                     shape [N]
#                 dist_an: pytorch Variable, distance between anchor and negative sample, 
#                     shape [N]
#             Returns:
#                 loss: pytorch Variable, with shape [1]
#         """
#         y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an, dist_ap, y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss
