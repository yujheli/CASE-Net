from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import cdist


def eval_metric(args, model, test_loader, query_loader):
    if args.eval_metric == 'rank':
        return rank_metric(args, model, test_loader, query_loader)

    elif args.eval_metric == 'mAP':
        pass


def extract_feature(dataloader, model):
    features_list = []
    infos = []
    for idx, batch in enumerate(dataloader):
        image = batch['image'].cuda()
        label = batch['label']
        camera_id = batch['camera_id']
        #_, latent_feature = model(image)
        latent_feature, _, _, _ = model(image)
        b = latent_feature.size()[0]
        latent_feature = latent_feature.view(b, -1)
        latent_feature = latent_feature.data.cpu().numpy()
        label = label.data.numpy()
        camera_id = camera_id.data.numpy()
        for i in range(b):
            features_list.append(latent_feature[i])
            infos.append((label[i], camera_id[i]))
    return features_list, infos


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


def rank_metric(args, model, test_loader, query_loader):
    test_features, test_infos = extract_feature(test_loader, model)
    query_features, query_infos = extract_feature(query_loader, model)

    match = []
    junk = []

    for _, (query_person, query_camera) in enumerate(query_infos):
        tmp_match = []
        tmp_junk = []
        for idx, (test_person, test_camera) in enumerate(test_infos):
            if test_person == query_person and query_camera != test_camera:
                tmp_match.append(idx)
            elif test_person == query_person or test_person < 0:
                tmp_junk.append(idx)
        match.append(tmp_match)
        junk.append(tmp_junk)

    dist_matrix = dist_metric(args, query_features, test_features)
    matrix_argsort = np.argsort(dist_matrix, axis=1)

    CMC = np.zeros([len(query_features), len(test_features)])    

    for idx in range(len(query_features)):
        counter = 0
        for i in range(len(test_features)):
            if matrix_argsort[idx][i] in junk[idx]:
                continue
            else:
                counter += 1
                if matrix_argsort[idx][i] in match[idx]:
                    CMC[idx, counter-1:] = 1
                if counter == args.rank:
                    break
    rank_1 = np.mean(CMC[:,0])
    return rank_1


def mAP_metric(model, dataloader):
    pass
