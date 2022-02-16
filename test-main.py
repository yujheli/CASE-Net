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
# from util.util import *
#from model.network import AdaptReID
from model.network import *
from model.discriminator import Discriminator, ACGAN
from loss.loss import ClassificationLoss, ReconstructionLoss
from loss.loss import TripletLoss
from parser.parser import ArgumentParser
from util.eval_utils import eval_metric
from util.util import *
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import config
import copy
from data.duke import Duke
from data.market import Market
from data.msmt import MSMT
from data.cuhk import CUHK
from data.viper import VIPER
from data.caviar import CAVIAR
from data.veri import VERI
import random
import torch.nn.functional as F

def init_model(args, use_cuda=True):

    if args.target_dataset == 'Duke':
        classifier_output_dim = config.DUKE_CLASS_NUM
    elif args.target_dataset == 'Market':
        classifier_output_dim = config.MARKET_CLASS_NUM
    elif args.target_dataset == 'MSMT':
        classifier_output_dim = config.MSMT_CLASS_NUM
    elif args.target_dataset == 'CUHK':
        classifier_output_dim = config.CUHK_CLASS_NUM
    elif args.target_dataset == 'VIPER':
        classifier_output_dim = config.VIPER_CLASS_NUM
    elif args.target_dataset == 'CAVIAR':
        classifier_output_dim = config.CAVIAR_CLASS_NUM
    elif args.target_dataset == 'VERI':
        classifier_output_dim = config.VERI_CLASS_NUM
    elif args.target_dataset == 'VRIC':
        classifier_output_dim = config.VRIC_CLASS_NUM    


    model = Sense_ReID(backbone='resnet-50',
                         use_cuda=use_cuda,
                         classifier_output_dim=classifier_output_dim)

    if args.extractor_path:
        print("Loading pre-trained extractor...")
        checkpoint = torch.load(args.extractor_path, map_location=lambda storage, loc: storage)['a']

#         model.structure_extractor.load_state_dict(checkpoint)
        for name, param in model.structure_extractor.state_dict().items():
            model.structure_extractor.state_dict()[name].copy_(checkpoint[name])

    if args.classifier_path:
        print("Loading pre-trained classifier...")
        checkpoint = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
#         for name, param in model.classifier.state_dict().items():
#             model.classifier.state_dict()[name].copy_(checkpoint[name])
        for name, param in model.structure_classifier.state_dict().items():
            model.structure_classifier.state_dict()[name].copy_(checkpoint[name])
            
    if args.generator_path:
        print("Loading pre-trained generator...")
        checkpoint = torch.load(args.generator_path, map_location=lambda storage, loc: storage)

        for name, param in model.generator.state_dict().items():
            model.generator.state_dict()[name].copy_(checkpoint[name])
    
    return model

def save_model(args, model, dis_model):

    s_extractor_path = os.path.join(args.model_dir, 'S_Extractor_{}.pth.tar'.format(args.source_dataset))
    c_extractor_path = os.path.join(args.model_dir, 'C_Extractor_{}.pth.tar'.format(args.source_dataset))
    classifier_path = os.path.join(args.model_dir, 'Classifier_{}.pth.tar'.format(args.source_dataset))
    generator_path = os.path.join(args.model_dir, 'Generator_{}.pth.tar'.format(args.source_dataset))
    discriminator_path = os.path.join(args.model_dir, 'Discriminator_{}.pth.tar'.format(args.source_dataset))

    torch.save(model.structure_extractor.state_dict(), s_extractor_path)
    torch.save(model.color_extractor.state_dict(), c_extractor_path)
    torch.save(model.structure_classifier.state_dict(), classifier_path)
    torch.save(model.generator.state_dict(), generator_path)
    torch.save(dis_model.state_dict(), discriminator_path)

    return

def init_target_data(args):

    if args.target_dataset == 'Duke':
        TargetData = Duke
    elif args.target_dataset == 'Market':
        TargetData = Market
    elif args.target_dataset == 'MSMT':
        TargetData = MSMT
    elif args.target_dataset == 'CUHK':
        TargetData = CUHK
    elif args.target_dataset == 'VIPER':
        TargetData = VIPER
    elif args.target_dataset == 'CAVIAR':
        TargetData = CAVIAR
    elif args.target_dataset == 'VERI':
        TargetData = VERI
    elif args.target_dataset == 'VRIC':
        TargetData = VRIC


    target_data = TargetData(mode='train',
                             transform=NormalizeImage(['image', 'rec_image']),
                             random_crop=args.random_crop,ds_factor=1,im_per_id=1)

    target_loader = DataLoader(target_data,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=True)

    return target_data, target_loader


def init_test_data(args):

    if args.target_dataset == 'Duke':
        TestData = Duke
    elif args.target_dataset == 'Market':
        TestData = Market
    elif args.target_dataset == 'MSMT':
        TestData = MSMT
    elif args.target_dataset == 'CUHK':
        TestData = CUHK
    elif args.target_dataset == 'VIPER':
        TestData = VIPER
    elif args.target_dataset == 'CAVIAR':
        TestData = CAVIAR
    elif args.target_dataset == 'VERI':
        TestData = VERI
    elif args.target_dataset == 'VRIC':
        TestData = VRIC

    test_data = TestData(mode='test',
                         transform=NormalizeImage(['image']),ds_factor=1,g_gray=False)

    test_loader = DataLoader(test_data,
                             batch_size=int(args.batch_size),
                             num_workers=args.num_workers,
                             pin_memory=True)

    return test_data, test_loader


def init_query_data(args):

    if args.target_dataset == 'Duke':
        QueryData = Duke
    elif args.target_dataset == 'Market':
        QueryData = Market
    elif args.target_dataset == 'MSMT':
        QueryData = MSMT
    elif args.target_dataset == 'CUHK':
        QueryData = CUHK
    elif args.target_dataset == 'VIPER':
        QueryData = VIPER
    elif args.target_dataset == 'CAVIAR':
        QueryData = CAVIAR
    elif args.target_dataset == 'VERI':
        QueryData = VERI
    elif args.target_dataset == 'VRIC':
        QueryData = VRIC


    query_data = QueryData(mode='query',
                           transform=NormalizeImage(['image']),ds_factor=1,q_gray=False)

    query_loader = DataLoader(query_data,
                              batch_size=int(args.batch_size),
                              num_workers=args.num_workers,
                              pin_memory=True)

    return query_data, query_loader

""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='train').parse()

def loss_triplet(pred, gt, use_cuda=True):
    criterion = TripletLoss(margin=config.GLOBAL_MARGIN).cuda()
#     criterion = ClassificationLoss(use_cuda=use_cuda)
#     loss = criterion(pred, gt)
    loss, prec = criterion(pred, gt.cuda())
    return loss

def loss_cls(pred, gt, use_cuda=True):
#     criterion = ClassificationLoss(use_cuda=use_cuda)
#     loss = criterion(pred, gt)
    loss = F.cross_entropy(pred, gt.cuda())
    return loss

def recon_criterion(input_, target):
    diff = input_ - target.detach()
    return torch.mean(torch.abs(diff[:]))

def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def main():

    """ GPU Settings """
    torch.cuda.set_device(args.gpu)
    use_cuda = torch.cuda.is_available()


    """ Initialize Model and Discriminators """
    model = init_model(args)
    
    dis_params = {'LAMBDA': 0.01,                   # the hyperparameter for the regularization term
                  'activ': 'lrelu',                   # activation function style [relu/lrelu/prelu/selu/tanh]
                  'dim': 32,                        # number of filters in the bottommost layer
                  'gan_type': 'lsgan',                # GAN loss [lsgan/nsgan]
                  'n_layer': 2,                     # number of layers in D
                  'n_res': 4,                       # number of layers in D
                  'non_local': 0,                   # number of non_local layers
                  'norm': 'none',                     # normalization layer [none/bn/in/ln]
                  'num_scales': 3,                  # number of scales
                  'pad_type': 'reflect'              # padding type [zero/reflect]
                 }
                  
    dis_model = MsImageDis(3, dis_params, fp16 = False).cuda() # discriminator for domain a
    
    if args.discriminator_path:
        print("Loading pre-trained discriminator...")
        checkpoint = torch.load(args.discriminator_path, map_location=lambda storage, loc: storage)

        for name, param in dis_model.state_dict().items():
            dis_model.state_dict()[name].copy_(checkpoint[name])



    """ Initialize Data """
#     source_data, source_loader = init_source_data(args)
#     source_iter = enumerate(source_loader)

    target_data, target_loader = init_target_data(args)
    target_iter = enumerate(target_loader)

    test_data, test_loader = init_test_data(args)

    query_data, query_loader = init_query_data(args)


   
        
    if True:
        print('Start evaluation...')
 
        model.eval()
            #rank1 = eval_metric(args, model, test_loader, query_loader)
        mAP, cmc, _, _ = eval_metric(args, model, test_loader, query_loader, re_rank=False)
        rank1, rank5, rank10, rank20 = cmc[[0,4,9,19]]

        print('Rank:', rank1, rank5, rank10, rank20)
        print('mAP:', mAP)


if __name__ == '__main__':
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    main()
