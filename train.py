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
from dataset.duke import Duke
from parser.parser import ArgumentParser
import config

""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='train').parse()

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
    model = AdaptReID(use_cuda=use_cuda)
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        for name, param in model.extractor.state_dict().items():
            model.extractor.state_dict()[name].copy_(checkpoint['state_dict']['extractor.' + name])    
        for name, param in model.classifier.state_dict().items():
            model.classifier.state_dict()[name].copy_(checkpoint['state_dict']['classifier.' + name])
        for name, param in model.decoder.state_dict().items():
            model.decoder.state_dict()[name].copy_(checkpoint['state_dict']['decoder.' + name])

    """ Initialize Discriminator """
    discriminator = Discriminator(use_cuda=use_cuda)
    if args.discriminator_path:
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

    if args.target_dataset == 'Duke':
        TargetData = Duke

    source_data = SourceData(csv_file=config.TRAIN_CSV_PATH,
                             dataset_path=config.TRAIN_DATA_PATH,
                             transform=NormalizeImage(['image']),
                             random_crop=args.random_crop)

    source_loader = DataLoader(source_data,
                               batch_size=args.batch_size,
                               shuffle=True, 
                               num_workers=args.num_workers,
                               pin_memory=True)

    source_iter = enumerate(source_loader)

    target_data = TargetData(csv_file=config.TRAIN_CSV_PATH,
                             dataset_path=config.TRAIN_DATA_PATH,
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
                           weight_decay=args.weight_decay)
    model_opt.zero_grad()

    discriminator_opt = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), 
                                   lr=args.learning_rate, 
                                   weight_decay=args.weight_decay)
    discriminator_opt.zero_grad()

    for step in range(args.num_steps):

        cls_loss = 0
        rec_loss = 0
        adv_loss = 0
        triplet_loss = 0

        model_opt.zero_grad()
        adjust_model_lr(model_opt, step)

        discriminator_opt.zero_grad()
        adjust_discriminator_lr(discriminator_opt, step)

        for idx in range(args.iter_size):

            """ Train Model and Fix Discriminator """
            for param in discriminator.parameters():
                param.requires_grad = False

            _, batch = source_iter.next()

            image = Variable(batch['image']).cuda()
            label = batch['label']

            features, cls_vector, reconstruct = model(image)

            #print(features)
            print('classification output:', cls_vector.size())
            print('reconstruct size:', reconstruct.size())
            print('cls[0] size:', cls_vector[0].size())
            print('cls[0]:', cls_vector[0])
            exit(-1)

if __name__ == '__main__':
    main()
