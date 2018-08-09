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
from model.discriminator import Discriminator
from loss.loss import ClassificationLoss
from loss.loss import ReconstructionLoss
from loss.loss import TripletLoss
from loss.loss import GlobalLoss
from loss.loss import LocalLoss
from data.duke import Duke
from data.market import Market
from data.msmt import MSMT
from data.cuhk import CUHK
from parser.parser import ArgumentParser
from util.eval_utils import eval_metric
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import config


""" Unnormalize """
inv_normalize = transforms.Normalize(
    mean=[-1.0, -1.0, -1.0],
    std=[2.0, 2.0, 2.0]
)



""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='train').parse()


def loss_rec(pred, gt, use_cuda=True):
    criterion = ReconstructionLoss(dist_metric='L1', 
                                   use_cuda=use_cuda)
    loss = criterion(pred, gt)
    return loss


def save_model(model, D_1=None, D_2=None):
    decoder_path = os.path.join(args.model_dir, 'Decoder_{}.pth.tar'.format(args.source_dataset))
    torch.save(model.decoder.state_dict(), decoder_path)
    return


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

    if args.classifier_path:
        print("Loading pre-trained classifier...")
        checkpoint = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
        for name, param in model.classifier.state_dict().items():
            model.classifier.state_dict()[name].copy_(checkpoint[name])    


    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)


    """ Initialize Source Data and Target Data """
    if args.source_dataset == 'Duke':
        SourceData = Duke
    elif args.source_dataset == 'Market':
        SourceData = Market
    elif args.source_dataset == 'MSMT':
        SourceData = MSMT
    elif args.source_dataset == 'CUHK':
        SourceData = CUHK
        
    if args.target_dataset == 'Duke':
        TargetData = Duke
        TestData = Duke
        QueryData = Duke
    elif args.target_dataset == 'Market':
        TargetData = Market
        TestData = Market
        QueryData = Market
    elif args.target_dataset == 'MSMT':
        TargetData = MSMT
        TestData = MSMT
        QueryData = MSMT
    elif args.target_dataset == 'CUHK':
        TargetData = CUHK
        TestData = CUHK
        QueryData = CUHK

    source_data = SourceData(mode='source',
                             transform=NormalizeImage(['image', 'rec_image']),
                             random_crop=args.random_crop)

    source_loader = DataLoader(source_data,
                               batch_size=args.batch_size,
                               shuffle=True, 
                               num_workers=args.num_workers,
                               pin_memory=True)

    source_iter = enumerate(source_loader)


    target_data = TargetData(mode='train',
                             transform=NormalizeImage(['image', 'rec_image']),
                             random_crop=args.random_crop)

    target_loader = DataLoader(target_data,
                               batch_size=args.batch_size,
                               shuffle=True, 
                               num_workers=args.num_workers,
                               pin_memory=True)

    target_iter = enumerate(target_loader)
    

    test_data = TestData(mode='test',
                         transform=NormalizeImage(['image']))

    test_loader = DataLoader(test_data,
                             batch_size=int(args.batch_size/2),
                             num_workers=args.num_workers,
                             pin_memory=True)
 

    query_data = QueryData(mode='query',
                           transform=NormalizeImage(['image']))

    query_loader = DataLoader(query_data,
                              batch_size=int(args.batch_size/2),
                              num_workers=args.num_workers,
                              pin_memory=True)


    """ Optimizer """
    model_opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate, 
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    model_opt.zero_grad()


    """ Initialize writer """
    writer = SummaryWriter()

    best_rank1 = 0


    """ Fix Extractor """
    for param in model.extractor.parameters():
        param.requires_grad = False

    """ Fix Classifier """
    for param in model.classifier.parameters():
        param.requires_grad = False


    """ Starts Training """
    for step in range(args.num_steps):
 
        model.train()

        rec_loss_value = 0

        model_opt.zero_grad()

        for idx in range(args.iter_size):

            """ Train Source Data """
            try:
                _, batch = next(source_iter)
            except:
                source_iter = enumerate(source_loader)
                _, batch = next(source_iter)

            image = batch['image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
            label = batch['label'].view(-1)
            rec_image = batch['rec_image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)


            latent_source, features_source, cls_source, rec_source, global_feature_source, local_feature_source = model(image) 

            loss = 0

            if args.rec_loss:
                rec_loss = loss_rec(pred=rec_source, gt=rec_image, use_cuda=use_cuda)
                rec_loss_value += rec_loss.data.cpu().numpy() / args.iter_size / 2.0
                loss += args.w_rec * rec_loss

            loss = loss / args.iter_size
            loss.backward()
            
 
            """ Write Images """
            if (step+1) % args.image_steps == 0:
                for i in range(len(rec_source)):
                    rec_source[i] = inv_normalize(rec_source[i])
                    image[i] = inv_normalize(image[i])

                writer.add_image('HR image', make_grid(image, nrow=16), step+1)
                writer.add_image('Reconstructed HR image', make_grid(rec_source, nrow=16), step+1)


            """ Train Target Data """
            try:
                _, batch = next(target_iter)
            except:
                target_iter = enumerate(target_loader)
                _, batch = next(target_iter)

            image = batch['image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
            label = batch['label'].view(-1)
            rec_image = batch['rec_image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)


            latent_target, features_target, cls_target, rec_target, global_feature_target, local_feature_target = model(image)

            loss = 0

            if args.rec_loss:
                rec_loss = loss_rec(pred=rec_target, gt=rec_image, use_cuda=use_cuda)
                rec_loss_value += rec_loss.data.cpu().numpy() / args.iter_size / 2.0
                loss += args.w_rec * rec_loss

            loss = loss / args.iter_size
            loss.backward()
            

            """ Write Images """
            if (step+1) % args.image_steps == 0:
                for i in range(len(rec_target)):
                    rec_target[i] = inv_normalize(rec_target[i])
                    image[i] = inv_normalize(image[i])

                writer.add_image('LR image', make_grid(image, nrow=16), step+1)
                writer.add_image('Reconstructed LR image', make_grid(rec_target, nrow=16), step+1)


        model_opt.step()

        print('[{0:6d}/{1:6d}] rec: {2:.6f}'.format(step+1, 
              args.num_steps, 
              rec_loss_value)) 
        
        """ Write Scalar """
        writer.add_scalar('Reconstruction Loss', rec_loss_value, step+1)
        
        if (step+1) % args.eval_steps == 0:
            print('Start evaluation...')
 
            model.eval()
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
                save_model(model)
                writer.add_scalar('Best Rank 1', best_rank1, (step+1)/args.eval_steps)

            print('Rank:', rank1, rank5, rank10, rank20)
            print('mAP:', mAP)
            print('Best rank1:', best_rank1)


if __name__ == '__main__':
    main()
