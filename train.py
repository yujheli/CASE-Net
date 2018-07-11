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
from loss.loss import ContrastiveLoss
from loss.loss import ReconstructionLoss
from data.duke import Duke
from data.market import Market
from parser.parser import ArgumentParser
import config


""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='train').parse()


def loss_rec(pred, gt, use_cuda=True):
    criterion = ReconstructionLoss(dist_metric='L1', use_cuda=use_cuda)
    loss = criterion(pred, gt)
    return loss


def loss_contra(pred, gt):
    criterion = ContrastiveLoss()
    loss = criterion(pred, gt)
    return loss


def loss_cls(pred, gt, use_cuda=True):
    criterion = ClassificationLoss(use_cuda=use_cuda)
    loss = criterion(pred, gt)
    return loss


def loss_adv(pred, gt):
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred, gt)
    return loss


def lr_poly(base_lr, idx, max_iter, power):
    return base_lr * ((1 - float(idx) / max_iter) ** (power))


def adjust_model_lr(optimizer, idx):
    lr = lr_poly(args.learning_rate, idx, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_discriminator_lr(optimizer, idx):
    lr = lr_poly(args.learning_rate, idx, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():

    """ GPU Settings """
    torch.cuda.set_device(args.gpu)
    use_cuda = torch.cuda.is_available()


    """ Initialize Model """
    if args.source_dataset == 'Duke':
        classifier_output_dim = 1812
    elif args.source_dataset == 'Market':
        classifier_output_dim = 1501
    
    model = AdaptReID(use_cuda=use_cuda,
                      classifier_output_dim=classifier_output_dim)
    if args.model_path:
        print("loading pretrained model...")
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        for name, param in model.extractor.state_dict().items():
            model.extractor.state_dict()[name].copy_(checkpoint['state_dict']['extractor.' + name])    
        for name, param in model.classifier.state_dict().items():
            model.classifier.state_dict()[name].copy_(checkpoint['state_dict']['classifier.' + name])
        for name, param in model.decoder.state_dict().items():
            model.decoder.state_dict()[name].copy_(checkpoint['state_dict']['decoder.' + name])


    """ Initialize Discriminator """
    discriminator = Discriminator(use_cuda=use_cuda,
                                  output_dim=1)
    if args.discriminator_path:
        print("loading pretrained discriminator...")
        checkpoint = torch.load(args.discriminator_path, map_location=lambda storage, loc: storage)
        for name, param in discriminator.state_dict().items():
            discriminator.state_dict()[name].copy_(checkpoint['state_dict']['discriminator.' + name])

    model.train()
    discriminator.train()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)


    """ Initialize Source Data and Target Data """
    if args.source_dataset == 'Duke':
        SourceData = Duke
    elif args.source_dataset == 'Market':
        SourceData = Market

    if args.target_dataset == 'Duke':
        TargetData = Duke
    elif args.target_dataset == 'Market':
        TargetData = Market

    source_data = SourceData(mode='source',
                             transform=NormalizeImage(['image']),
                             random_crop=args.random_crop)

    source_loader = DataLoader(source_data,
                               batch_size=args.batch_size,
                               shuffle=True, 
                               num_workers=args.num_workers,
                               pin_memory=True)

    source_iter = enumerate(source_loader)

    target_data = TargetData(mode='train',
                             transform=NormalizeImage(['image']),
                             random_crop=args.random_crop)

    target_loader = DataLoader(target_data,
                               batch_size=args.batch_size,
                               shuffle=True, 
                               num_workers=args.num_workers,
                               pin_memory=True)

    target_iter = enumerate(target_loader)


    """ Initialize Optimizer """
    model_opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.learning_rate, 
                           betas=(0.9, 0.99))
    model_opt.zero_grad()

    discriminator_opt = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), 
                                   lr=args.learning_rate, 
                                   betas=(0.9, 0.99))
    discriminator_opt.zero_grad()

    source_label = 0
    target_label = 1

    """ Fix Extractor """
    for param in model.extractor.parameters():
        param.requires_grad = False

    """ Fix Decoder """
    for param in model.decoder.parameters():
        param.requires_grad = False

    for step in range(args.num_steps):

        cls_loss_value = 0
        rec_loss_value = 0
        contra_loss_value = 0
        adv_target_loss_value = 0
        dis_loss_value = 0 # Discriminator's loss

        model_opt.zero_grad()
        adjust_model_lr(model_opt, step)

        discriminator_opt.zero_grad()
        adjust_discriminator_lr(discriminator_opt, step)

        for idx in range(args.iter_size):

            """ Train Model and Fix Discriminator """
            for param in discriminator.parameters():
                param.requires_grad = False

            """ Train Source Data """
            try:
                _, batch = source_iter.next()
            except:
                source_iter = enumerate(source_loader)
                _, batch = source_iter.next()

            image = Variable(batch['image']).cuda()
            label = batch['label']

            latent_source, extracted_source, cls_source, rec_source = model(image)

            loss = 0

            if args.cls_loss:
                cls_loss = loss_cls(pred=cls_source, gt=label, use_cuda=use_cuda)
                loss += args.w_cls * cls_loss
                cls_loss_value += cls_loss.data.cpu().numpy() / args.iter_size
    
            if args.contra_loss:
                contra_loss = loss_contra(pred=latent_source, gt=label)
                loss += args.w_contra * contra_loss
                contra_loss_value += contra_loss.data.cpu().numpy() / args.iter_size

            if args.rec_loss:
                rec_loss = loss_rec(pred=rec_source, gt=image, use_cuda=use_cuda)
                loss += args.w_rec * rec_loss
                rec_loss_value += rec_loss.data.cpu().numpy() / args.iter_size

            loss = loss / args.iter_size
            loss.backward()

            '''
            """ Train Target Data """
            _, batch = target_iter.next()

            image = Variable(batch['image']).cuda()
            label = batch['label']

            latent_target, extracted_target, cls_target, rec_target = model(image)

            D_output = discriminator(extracted_target)

            tensor = Variable(torch.FloatTensor(D_output.data.size()).fill_(source_label)).cuda()
            if args.adv_loss:
                adv_loss = loss_adv(pred=D_output, gt=tensor)
                adv_target_loss_value += adv_loss.data.cpu().numpy() / args.iter_size

                loss = args.w_adv * adv_loss
                loss.backward()


            """ Train Discriminator """
            for param in discriminator.parameters():
                param.requires_grad = True

            """ Train with Source Data """
            extracted_source = extracted_source.detach()
            
            D_output = discriminator(extracted_source)

            tensor = Variable(torch.FloatTensor(D_output.data.size()).fill_(source_label)).cuda()
            if args.dis_loss:
                dis_loss = loss_adv(pred=D_output, gt=tensor) / args.iter_size / 2
                dis_loss_value += dis_loss.data.cpu().numpy()

                loss = args.w_dis * dis_loss
                loss.backward()


            """ Train with Target Data """
            extracted_target = extracted_target.detach()
 
            D_output = discriminator(extracted_target)

            tensor = Variable(torch.FloatTensor(D_output.data.size()).fill_(target_label)).cuda()
            if args.dis_loss:
                dis_loss = loss_adv(pred=D_output, gt=tensor) / args.iter_size / 2
                dis_loss_value += dis_loss.data.cpu().numpy()

                loss = args.w_dis * dis_loss
                loss.backward()
            '''

        model_opt.step()
        #discriminator_opt.step()

        print('[{0:4d}/{1:4d}] cls: {2:.3f} rec: {3:.3f} contra: {4:.3f} adv: {5:.3f} dis: {6:.3f}'.format(step+1, 
            args.num_steps, cls_loss_value, rec_loss_value, contra_loss_value, adv_target_loss_value, dis_loss_value))

        if step >= args.num_steps_stop - 1:
            print('Saving model...')
            model_path = os.path.join(args.model_dir, 'Model_Duke_{}.pth.tar'.format(args.num_steps))
            discriminator_path = os.path.join(args.model_dir, 'Discriminator_Duke_{}.pth.tar'.format(args.num_steps))
            torch.save(model.state_dict(), model_path)
            torch.save(discriminator.state_dict(), discriminator_path)

        if (step+1) % args.save_steps == 0:
            print('Saving model...')
            model_path = os.path.join(args.model_dir, 'Model_Duke_{}.pth.tar'.format(step+1))
            discriminator_path = os.path.join(args.model_dir, 'Discriminator_Duke_{}.pth.tar'.format(step+1))
            torch.save(model.state_dict(), model_path)
            torch.save(discriminator.state_dict(), discriminator_path)

if __name__ == '__main__':
    main()
