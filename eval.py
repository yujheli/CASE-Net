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
from model.network import Extractor
from data.duke import Duke
from data.market import Market
from data.msmt import MSMT
from data.cuhk import CUHK
from parser.parser import ArgumentParser
from util.eval_util import eval_metric
import config


""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='eval').parse()


def main():

    """ GPU Settings """
    torch.cuda.set_device(args.gpu)
    use_cuda = torch.cuda.is_available()


    """ Initialize Model """
    model = Extractor(backbone=args.backbone, 
                      skip_connection=False)

    if use_cuda:
        model = model.cuda()

    if args.model_path:
        print("Loading from trained model...")
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        for name, param in model.state_dict().items():
            model.state_dict()[name].copy_(checkpoint['extractor.' + name])    

    if args.extractor_path:
        print("Loading from trained extractor...")
        checkpoint = torch.load(args.extractor_path, map_location=lambda storage, loc: storage)
        for name, param in model.state_dict().items():
            model.state_dict()[name].copy_(checkpoint['extractor.' + name])    


    """ Initialize Test Data and Target Data """
    if args.dataset == 'Duke':
        TestData = Duke
        QueryData = Duke
    elif args.dataset == 'Market':
        TestData = Market
        QueryData = Market
    elif args.dataset == 'MSMT':
        TestData = MSMT
        QueryData = MSMT
    elif args.dataset == 'CUHK':
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
    model.eval()

    eval_metric(args, model, test_loader, query_loader)

if __name__ == '__main__':
    main()
