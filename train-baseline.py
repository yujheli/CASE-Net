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
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import config

from data.duke import Duke
from data.market import Market
from data.msmt import MSMT
from data.cuhk import CUHK
from data.viper import VIPER
from data.caviar import CAVIAR
from data.veri import VERI

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


    model = Baseline_ReID(backbone='resnet-50',
                         use_cuda=use_cuda,
                         classifier_output_dim=classifier_output_dim)

    if args.extractor_path:
        print("Loading pre-trained extractor...")
        checkpoint = torch.load(args.extractor_path, map_location=lambda storage, loc: storage)
        for name, param in model.extractor.state_dict().items():
            model.extractor.state_dict()[name].copy_(checkpoint[name])

    if args.classifier_path:
        print("Loading pre-trained classifier...")
        checkpoint = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
        for name, param in model.classifier.state_dict().items():
            model.classifier.state_dict()[name].copy_(checkpoint[name])

    return model

def save_model(args, model):

    extractor_path = os.path.join(args.model_dir, 'Extractor_{}.pth.tar'.format(args.source_dataset))
    classifier_path = os.path.join(args.model_dir, 'Classifier_{}.pth.tar'.format(args.source_dataset))

    torch.save(model.extractor.state_dict(), extractor_path)
    torch.save(model.classifier.state_dict(), classifier_path)

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
                             random_crop=args.random_crop,ds_factor=1)

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
                           transform=NormalizeImage(['image']),ds_factor=1,q_gray=True)

    query_loader = DataLoader(query_data,
                              batch_size=int(args.batch_size),
                              num_workers=args.num_workers,
                              pin_memory=True)

    return query_data, query_loader


def init_model_optim(args, model):
    
    model_opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    model_opt.zero_grad()

    return model_opt

""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='train').parse()

# def loss_triplet(global_feature, local_feature, label, normalize=True):
#     criterion = TripletLoss(margin=config.GLOBAL_MARGIN)
#     global_loss, pos_inds, neg_inds = GlobalLoss(criterion, 
#                                                  global_feature,
#                                                  label.cuda(),
#                                                  normalize_feature=normalize)

#     criterion = TripletLoss(margin=config.LOCAL_MARGIN)
#     local_loss = LocalLoss(criterion, 
#                            local_feature,
#                            pos_inds,
#                            neg_inds,
#                            label.cuda(),
#                            normalize_feature=normalize)

#     return global_loss, local_loss

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



def main():

    """ GPU Settings """
    torch.cuda.set_device(args.gpu)
    use_cuda = torch.cuda.is_available()


    """ Initialize Model and Discriminators """
    model = init_model(args)

#     D_resolution = init_resolution_D(args)

#     D_ACGAN = init_ACGAN(args)


    """ Initialize Data """
#     source_data, source_loader = init_source_data(args)
#     source_iter = enumerate(source_loader)

    target_data, target_loader = init_target_data(args)
    target_iter = enumerate(target_loader)

    test_data, test_loader = init_test_data(args)

    query_data, query_loader = init_query_data(args)


    """ Initialize Optimizers """
    model_opt = init_model_optim(args, model)


    """ Initialize Writer """
    writer = SummaryWriter()

    best_rank1 = 0
    

    """ Start Training """
    for step in range(args.num_steps):
 
        model.train()
#         D_resolution.train()
#         D_ACGAN.train()
        
        diff_loss_value = 0
        
        cls_loss_value = 0
        rec_loss_value = 0
 
        global_loss_value = 0
        local_loss_value = 0

        model_opt.zero_grad()
#        

        """ Train Target Data """
        try:
            _, batch = next(target_iter)
        except:
            target_iter = enumerate(target_loader)
            _, batch = next(target_iter)

        image = batch['image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        label = batch['label'].view(-1)
#         print(label)
#         rec_image = batch['rec_image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)


        """ Model Return """
        target_dict = model(image)
#         print(target_dict['cls_vector'].data.cpu().numpy().shape)

        """ Target Training Loss """
        loss = 0
        if args.cls_loss:
            cls_loss = loss_cls(pred=target_dict['cls_vector'], 
                                gt=label, 
                                use_cuda=True)

            cls_loss_value += cls_loss.data.cpu().numpy() / args.iter_size
            loss += args.w_cls * cls_loss

        if args.triplet_loss:
#             global_loss, local_loss = loss_triplet(global_feature=target_dict['global_feature'],
#                                                    local_feature=target_dict['local_feature'],
#                                                    label=label)
            global_loss = loss_triplet(pred=target_dict['global_feature'],gt=label)

            global_loss_value += global_loss.data.cpu().numpy() / args.iter_size
#             local_loss_value += local_loss.data.cpu().numpy() / args.iter_size

            loss += args.w_global * global_loss
#             loss += args.w_local * local_loss


        loss = loss / args.iter_size
        loss.backward()
        
        model_opt.step()


        print_string = '[{:6d}/{:6d}]'.format(step+1, args.num_steps)
        
        if args.cls_loss:
            print_string += ' cls: {:.6f}'.format(cls_loss_value)

#         if args.rec_loss:
#             print_string += ' rec: {:.6f}'.format(rec_loss_value)

        if args.triplet_loss:
            print_string += ' global: {:.6f} local: {:.6f}'.format(global_loss_value, local_loss_value)

        print(print_string)

        
        """ Write Scalar """
        writer.add_scalar('Classification Loss', cls_loss_value, step+1)
#         writer.add_scalar('Reconstruction Loss', rec_loss_value, step+1)
#         writer.add_scalar('Adversarial Loss', D_resolution_adv_loss_value, step+1)
#         writer.add_scalar('Discriminator Loss', D_resolution_dis_loss_value, step+1)
#         writer.add_scalar('ACGAN Adversarial Loss', D_ACGAN_adv_loss_value, step+1)
#         writer.add_scalar('ACGAN Discriminator Loss', D_ACGAN_dis_loss_value, step+1)
#         writer.add_scalar('ACGAN Classification Loss', D_ACGAN_cls_loss_value, step+1)
        writer.add_scalar('Global Triplet Loss', global_loss_value, step+1)
        writer.add_scalar('Local Triplet Loss', local_loss_value, step+1)
#         writer.add_scalar('KLD Loss', KL_loss_value, step+1)
#         writer.add_scalar('GP Loss', GP_loss_value, step+1)
#         writer.add_scalar('Diff Loss', diff_loss_value, step+1)

        
        if (step+1) % args.eval_steps == 0:
            print('Start evaluation...')
 
            model.eval()
            #rank1 = eval_metric(args, model, test_loader, query_loader)
            mAP, cmc, _, _ = eval_metric(args, model, test_loader, query_loader, re_rank=False)
            rank1, rank5, rank10, rank20 = cmc[[0,4,9,19]]
            
            writer.add_scalar('Rank 1', rank1, (step+1)/args.eval_steps)
            writer.add_scalar('Rank 5', rank5, (step+1)/args.eval_steps)
            writer.add_scalar('Rank 10', rank10, (step+1)/args.eval_steps)
            writer.add_scalar('Rank 20', rank20, (step+1)/args.eval_steps)
            writer.add_scalar('mAP', mAP, (step+1)/args.eval_steps)

            if rank1 >= best_rank1:
                best_rank1 = rank1
                print('Saving model...')
                save_model(args, model) # To save model
                writer.add_scalar('Best Rank 1', best_rank1, (step+1)/args.eval_steps)

            print('Rank:', rank1, rank5, rank10, rank20)
            print('mAP:', mAP)
            print('Best rank1:', best_rank1)


if __name__ == '__main__':
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    main()
