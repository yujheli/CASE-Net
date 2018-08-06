from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from util.dataloader import DataLoader
from util.normalize import NormalizeImage
from util.torch_util import save_checkpoint
from model.network import AdaptReID
from data.duke import Duke
from data.market import Market
from data.msmt import MSMT
from data.cuhk import CUHK
from parser.parser import ArgumentParser
from util.eval_utils import eval_metric
import config


""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='train').parse()


def main():

    """ GPU Settings """
    torch.cuda.set_device(args.gpu)
    use_cuda = torch.cuda.is_available()


    """ Initialize Model """
    if args.source_dataset == 'Duke':
        classifier_output_dim = config.DUKE_CLASS_NUM
    elif args.source_dataset == 'Market':
        classifier_output_dim = config.MARKET_CLASS_NUM
    elif args.source_dataset == 'MSMT':
        classifier_output_dim = config.MSMT_CLASS_NUM
    elif args.source_dataset == 'CUHK':
        classifier_output_dim = config.CUHK_CLASS_NUM
    
    model = AdaptReID(backbone='resnet-50',
                      use_cuda=use_cuda,
                      classifier_output_dim=classifier_output_dim)

    if args.extractor_path:
        print("Loading pre-trained extractor...")
        checkpoint = torch.load(args.extractor_path, map_location=lambda storage, loc: storage)
        for name, param in model.extractor.state_dict().items():
            model.extractor.state_dict()[name].copy_(checkpoint[name])    


    if args.target_dataset == 'Duke':
        TestData = Duke
        QueryData = Duke
    elif args.target_dataset == 'Market':
        TestData = Market
        QueryData = Market
    elif args.target_dataset == 'MSMT':
        TestData = MSMT
        QueryData = MSMT
    elif args.target_dataset == 'CUHK':
        TestData = CUHK
        QueryData = CUHK


    test_data = TestData(mode='test',
                         transform=NormalizeImage(['image']))

    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True)
 

    query_data = QueryData(mode='query',
                           transform=NormalizeImage(['image']))

    query_loader = DataLoader(query_data,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)

    print('Start evaluation...')
 
    model.eval()
    mAP, cmc, rerank_mAP, rerank_cmc = eval_metric(args, model, test_loader, query_loader, re_rank=True)

    rank1, rank5, rank10, rank20 = cmc[[0,4,9,19]]
    
    re_rank1, re_rank5, re_rank10, re_rank20 = rerank_cmc[[0,4,9,19]]

    print('Rank 1:', rank1)
    print('Rank 5:', rank5)
    print('Rank 10:', rank10)
    print('Rank 20:', rank20)
    print('mAP:', mAP)

    print('Re-Rank 1:', re_rank1)
    print('Re-Rank 5:', re_rank5)
    print('Re-Rank 10:', re_rank10)
    print('Re-Rank 20:', re_rank20)
    print('Re-Rank mAP:', rerank_mAP)


if __name__ == '__main__':
    main()
