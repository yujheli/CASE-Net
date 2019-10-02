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

def eval_metric(args, model, test_loader, query_loader, re_rank=True):
    return rank_metric(args, model, test_loader, query_loader, normalize_feat=True,\
        use_local_distance=False,\
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
        dic = model(image)
        global_feat = dic['global_feature']
        local_feat = dic['local_feature']
        
        
        global_feat = global_feat.data.cpu().numpy()
        local_feat = local_feat.data.cpu().numpy()
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
    local_feats = np.concatenate(local_feats)
    ids = np.hstack(ids)
    cams = np.hstack(cams)
    if normalize_feat:
        global_feats = normalize(global_feats, axis=1)
        local_feats = normalize(local_feats, axis=-1)

            
            
    return global_feats, local_feats, ids, cams

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

    return mAP, cmc_scores, mAP_rerank, cmc_scores_rerank



