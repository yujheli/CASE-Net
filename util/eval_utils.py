from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import cdist
from util.re_ranking import re_ranking
from util.metric import cmc, mean_ap
from util.distance import normalize, compute_dist, local_dist, low_memory_matrix_op 
from util.utils import measure_time
import time
import sys

import numpy as np
from sklearn import metrics as sk_metrics

class PersonReIDMAP:
    '''
    Compute Rank@k and mean Average Precision (mAP) scores
    Used for Person ReID
    Test on MarKet and Duke
    '''

    def __init__(self, query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label, dist):
        '''
        :param query_feature: np.array, bs * feature_dim
        :param query_cam: np.array, 1d
        :param query_label: np.array, 1d
        :param gallery_feature: np.array, gallery_size * feature_dim
        :param gallery_cam: np.array, 1d
        :param gallery_label: np.array, 1d
        '''

        self.query_feature = query_feature
        self.query_cam = query_cam
        self.query_label = query_label
        self.gallery_feature = gallery_feature
        self.gallery_cam = gallery_cam
        self.gallery_label = gallery_label

        assert dist in ['cosine', 'euclidean']
        self.dist = dist

        # normalize feature for fast cosine computation
        if self.dist == 'cosine':
            self.query_feature = self.normalize(self.query_feature)
            self.gallery_feature = self.normalize(self.gallery_feature)

        APs = []
        CMC = []
        for i in range(len(query_label)):
            AP, cmc = self.evaluate(self.query_feature[i], self.query_cam[i], self.query_label[i],
                                    self.gallery_feature, self.gallery_cam, self.gallery_label)
            APs.append(AP)
            CMC.append(cmc)
            # print('{}/{}'.format(i, len(query_label)))

        self.APs = np.array(APs)
        self.mAP = np.mean(self.APs)

        min_len = 99999999
        for cmc in CMC:
            if len(cmc) < min_len:
                min_len = len(cmc)
        for i, cmc in enumerate(CMC):
            CMC[i] = cmc[0: min_len]
        self.CMC = np.mean(np.array(CMC), axis=0)

    def compute_AP(self, index, good_index):
        '''
        :param index: np.array, 1d
        :param good_index: np.array, 1d
        :return:
        '''

        num_good = len(good_index)
        hit = np.in1d(index, good_index)
        index_hit = np.argwhere(hit == True).flatten()

        if len(index_hit) == 0:
            AP = 0
            cmc = np.zeros([len(index)])
        else:
            precision = []
            for i in range(num_good):
                precision.append(float(i+1) / float((index_hit[i]+1)))
            AP = np.mean(np.array(precision))
            cmc = np.zeros([len(index)])
            cmc[index_hit[0]: ] = 1

        return AP, cmc

    def evaluate(self, query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label):
        '''
        :param query_feature: np.array, 1d
        :param query_cam: int
        :param query_label: int
        :param gallery_feature: np.array, 2d, gallerys_size * feature_dim
        :param gallery_cam: np.array, 1d
        :param gallery_label: np.array, 1d
        :return:
        '''

        # cosine score
        if self.dist is 'cosine':
            # feature has been normalize during intialization
            score = np.matmul(query_feature, gallery_feature.transpose())
            index = np.argsort(score)[::-1]
        elif self.dist is 'euclidean':
            score = self.l2(query_feature.reshape([1, -1]), gallery_feature)
            index = np.argsort(score.reshape([-1]))

        junk_index_1 = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam == gallery_cam))
        junk_index_2 = np.argwhere(gallery_label == -1)
        junk_index = np.append(junk_index_1, junk_index_2)

        good_index = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam != gallery_cam))
        index_wo_junk = self.notin1d(index, junk_index)

        return self.compute_AP(index_wo_junk, good_index)

    def in1d(self, array1, array2, invert=False):
        '''
        :param set1: np.array, 1d
        :param set2: np.array, 1d
        :return:
        '''

        mask = np.in1d(array1, array2, invert=invert)
        return array1[mask]

    def notin1d(self, array1, array2):

        return self.in1d(array1, array2, invert=True)

    def normalize(self, x):
        norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
        return x / norm

    def cosine_dist(self, x, y):
        return sk_metrics.pairwise.cosine_distances(x, y)

    def euclidean_dist(self, x, y):
        return sk_metrics.pairwise.euclidean_distances(x, y)




def eval_metric(args, model, test_loader, query_loader, re_rank=True, use_local=False):
    return rank_metric2(args, model, test_loader, query_loader, normalize_feat=True,\
        use_local_distance=use_local,\
        to_re_rank=re_rank,\
        pool_type='average')
    if args.eval_metric == 'rank':
        return rank_metric(args, model, test_loader, query_loader)

    elif args.eval_metric == 'mAP':
        pass


def dist_metric(args, query_features, test_features):
    if args.dist_metric == 'L2':
        dist = 'euclidean'
    elif args.dist_metric == 'L1':
        dist = 'hamming'
    elif args.dist_metric == 'cosine':
        dist = 'cosine'
    elif args.dist_metric == 'correlation':
        dist = 'correlation'
    matrix = cdist(query_features, test_features, dist)
    return matrix

def mAP_metric(model, dataloader):
    pass

"""New added"""

def extract_feat(dataloader, model, normalize_feat):
    """Extract the features of the whole image set.
    Args:
        normalize_feat: True or False, whether to normalize global and local 
            feature to unit length
    Returns:
        global_feats: numpy array with shape [N, C]
        local_feats: numpy array with shape [N, H, c]
        ids: numpy array with shape [N]
        cams: numpy array with shape [N]
        im_names: numpy array with shape [N]
        marks: numpy array with shape [N]
    """
    
#     def extract_feature(dataloader, model):
    features_list = []
    infos = []
    global_feats, local_feats, ids, cams = [], [], [], []
        
    printed = False
    st = time.time()
    last_time = time.time()
    
    
    
    for idx, batch in enumerate(dataloader):
        image = batch['image'].cuda()
        label = batch['label']
        camera_id = batch['camera_id']
        #_, _, _, _, global_feat, local_feat, _, _ = model(image)
#         local, feat = model.extractor(image)
        return_dict = model(image)
#         return_dict = {'cls_vectors': y, 
#                        'global_feature': feat,
#                        'local_feature': local_f}
        global_feat = return_dict['global_feature']
        local_feat = return_dict['local_feature']
        
        
        global_feat = global_feat.data.cpu().numpy()
        local_feat = local_feat.view(local_feat.size(0),-1).data.cpu().numpy()
#         print(local_feat.shape)
        label = label.data.numpy()
#         print("camera_id",camera_id)
#         camera_id = camera_id.data.numpy()
        
#         print('global_type',type(global_feats))
        global_feats.append(global_feat)
        local_feats.append(local_feat)
        ids.append(label)
        cams.append(camera_id)
        
#         b = latent_feature.size()[0]
#         latent_feature = latent_feature.view(b, -1)
        
        
        if (idx+1) % 20 == 0:
            if not printed:
                printed = True
            else:
                # Clean the current line
                sys.stdout.write("\033[F\033[K")
            print('{}/{} batches done, +{:.2f}s, total {:.2f}s'
                        .format(idx, len(dataloader),
                                        time.time() - last_time, time.time() - st))
            last_time = time.time()
      
    global_feats = np.vstack(global_feats)
    local_feats = np.vstack(local_feats)
    ids = np.hstack(ids)
    cams = np.hstack(cams)
    if normalize_feat:
        global_feats = normalize(global_feats, axis=1)
        local_feats = normalize(local_feats, axis=-1)

            
    return global_feats, local_feats, ids, cams


def rank_metric2(args, model, test_loader, query_loader, normalize_feat=True,
        use_local_distance=False,
        to_re_rank=True,
        pool_type='average'):

    mAP, cmc_scores, mAP_rerank, cmc_scores_rerank = None, None, None, None
    
    
    with measure_time('Extracting feature...'):
        global_feats_test, local_feats_test, ids_test, cams_test = \
            extract_feat(test_loader, model, normalize_feat)
        global_feats_query, local_feats_query, ids_query, cams_query = \
            extract_feat(query_loader, model, normalize_feat)
        
    result_global = PersonReIDMAP(
        global_feats_query, cams_query, ids_query,
        global_feats_test, cams_test, ids_test, dist='cosine')
    
    if use_local_distance==True:
        result_local = PersonReIDMAP(
        local_feats_query, cams_query, ids_query,
        local_feats_test, cams_test, ids_test, dist='cosine')
        return result_global.mAP, np.array(result_global.CMC[0: 150]), mAP_rerank, cmc_scores_rerank, result_local.mAP, np.array(result_local.CMC[0: 150]),

        

    return result_global.mAP, list(result_global.CMC[0: 150]), mAP_rerank, cmc_scores_rerank


def eval_map_cmc(
        q_g_dist,
        q_ids=None, g_ids=None,
        q_cams=None, g_cams=None,
        separate_camera_set=None,
        single_gallery_shot=None,
        first_match_break=None,
        topk=None):
    """Compute CMC and mAP.
    Args:
        q_g_dist: numpy array with shape [num_query, num_gallery], the 
            pairwise distance between query and gallery samples
    Returns:
        mAP: numpy array with shape [num_query], the AP averaged across query 
            samples
        cmc_scores: numpy array with shape [topk], the cmc curve 
            averaged across query samples
    """
    # Compute mean AP
    mAP = mean_ap(
        distmat=q_g_dist,
        query_ids=q_ids, gallery_ids=g_ids,
        query_cams=q_cams, gallery_cams=g_cams)
    # Compute CMC scores
    cmc_scores = cmc(
        distmat=q_g_dist,
        query_ids=q_ids, gallery_ids=g_ids,
        query_cams=q_cams, gallery_cams=g_cams,
        separate_camera_set=separate_camera_set,
        single_gallery_shot=single_gallery_shot,
        first_match_break=first_match_break,
        topk=topk)
    
#     print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
#                 .format(mAP, *cmc_scores[[0, 4, 9]]))
    return mAP, cmc_scores


def rank_metric(args, model, test_loader, query_loader, normalize_feat=True,
        use_local_distance=False,
        to_re_rank=True,
        pool_type='average'):

    """Evaluate using metric CMC and mAP.
    Args:
        normalize_feat: whether to normalize features before computing distance
        use_local_distance: whether to use local distance
        to_re_rank: whether to also report re-ranking scores
        pool_type: 'average' or 'max', only for multi-query case
    """
    mAP, cmc_scores, mAP_rerank, cmc_scores_rerank = None, None, None, None
    
    with measure_time('Extracting feature...'):
        global_feats_test, local_feats_test, ids_test, cams_test = \
            extract_feat(test_loader, model, normalize_feat)
        global_feats_query, local_feats_query, ids_query, cams_query = \
            extract_feat(query_loader, model, normalize_feat)

    def compute_score(dist_mat):
        mAP, cmc_scores = eval_map_cmc(
            q_g_dist=dist_mat,
            q_ids=ids_query, g_ids=ids_test,
            q_cams=cams_query, g_cams=cams_test,
            separate_camera_set=False, #self.separate_camera_set,
            single_gallery_shot=False, #self.single_gallery_shot,
            first_match_break=True, #self.first_match_break,
            topk=100)
        return mAP, cmc_scores

    # A helper function just for avoiding code duplication.
    def low_memory_local_dist(x, y):
        with measure_time('Computing local distance...'):
            x_num_splits = int(len(x) / 200) + 1
            y_num_splits = int(len(y) / 200) + 1
            z = low_memory_matrix_op(
                local_dist, x, y, 0, 0, x_num_splits, y_num_splits, verbose=True)
        return z

    ###################
    # Global Distance #
    ###################
    if args.dist_metric == 'L2':
        dist_type = 'euclidean'
    else:
        dist_type = 'cosine'

    with measure_time('Computing global distance...'):
        # query-gallery distance using global distance
        global_q_g_dist = compute_dist(
            global_feats_query, global_feats_test, type='euclidean')

    with measure_time('Computing scores for Global Distance...'):
        mAP, cmc_scores = compute_score(global_q_g_dist)
    
    ###################
    # Local Distance #
    ###################
    
    if use_local_distance==True:
        
        with measure_time('Computing local distance...'):
            # query-gallery distance using global distance
            local_q_g_dist = compute_dist(
                local_feats_query, local_feats_test, type='euclidean')

        with measure_time('Computing scores for Local Distance...'):
            local_mAP, local_cmc_scores = compute_score(local_q_g_dist)
        
        
    if to_re_rank:
        with measure_time('Re-ranking...'):
            # query-query distance using global distance
            global_q_q_dist = compute_dist(
                global_feats_query, global_feats_query, type='euclidean') #or cosine

            # gallery-gallery distance using global distance
            global_g_g_dist = compute_dist(
                global_feats_test, global_feats_test, type='euclidean')

            # re-ranked global query-gallery distance
            re_r_global_q_g_dist = re_ranking(
                global_q_g_dist, global_q_q_dist, global_g_g_dist)

        with measure_time('Computing scores for re-ranked Global Distance...'):
            mAP_rerank, cmc_scores_rerank = compute_score(re_r_global_q_g_dist)
    
    if use_local_distance==True:   
        return mAP, cmc_scores, mAP_rerank, cmc_scores_rerank, local_mAP, local_cmc_scores
    
    return mAP, cmc_scores, mAP_rerank, cmc_scores_rerank



