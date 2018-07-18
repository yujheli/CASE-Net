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
from data.msmt import MSMT
from data.cuhk import CUHK
from parser.parser import ArgumentParser
from util.eval_util import eval_metric
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid, save_image # Newly added
from torchvision import transforms # Newly added
import config

""" Unnormalize """
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='train').parse()


def loss_rec(pred, gt, use_cuda=True):
    criterion = ReconstructionLoss(dist_metric='L1', 
                                   use_cuda=use_cuda)
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
    return


def adjust_discriminator_lr(optimizer, idx):
    lr = lr_poly(args.learning_rate, idx, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return


def save_model(model, discriminator, step):
    model_path = os.path.join(args.model_dir, 'Model_{}_{}.pth.tar'.format(args.source_dataset, step))
    extractor_path = os.path.join(args.model_dir, 'Extractor_{}_{}.pth.tar'.format(args.source_dataset, step))
    decoder_path = os.path.join(args.model_dir, 'Decoder_{}_{}.pth.tar'.format(args.source_dataset, step))
    classifier_path = os.path.join(args.model_dir, 'Classifier_{}_{}.pth.tar'.format(args.source_dataset, step))
    discriminator_path = os.path.join(args.model_dir, 'Discriminator_{}_{}.pth.tar'.format(args.source_dataset, step))
            
    torch.save(model.state_dict(), model_path)
    torch.save(model.extractor.state_dict(), extractor_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    torch.save(model.classifier.state_dict(), classifier_path)
    torch.save(discriminator.state_dict(), discriminator_path)
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
    
    #model = AdaptReID(backbone=args.backbone,
    model = AdaptReID(backbone='resnet-50',
                      use_cuda=use_cuda,
                      classifier_output_dim=classifier_output_dim)

    if args.model_path:
        print("Loading pre-trained model...")
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        for name, param in model.extractor.state_dict().items():
            model.extractor.state_dict()[name].copy_(checkpoint['extractor.' + name])    
        for name, param in model.classifier.state_dict().items():
            model.classifier.state_dict()[name].copy_(checkpoint['classifier.' + name])
        for name, param in model.decoder.state_dict().items():
            model.decoder.state_dict()[name].copy_(checkpoint['decoder.' + name])

    if args.extractor_path:
        print("Loading pre-trained extractor...")
        checkpoint = torch.load(args.extractor_path, map_location=lambda storage, loc: storage)
        for name, param in model.extractor.state_dict().items():
            model.extractor.state_dict()[name].copy_(checkpoint['extractor.' + name])    

    if args.decoder_path:
        print("Loading pre-trained decoder...")
        checkpoint = torch.load(args.decoder_path, map_location=lambda storage, loc: storage)
        for name, param in model.decoder.state_dict().items():
            model.decoder.state_dict()[name].copy_(checkpoint['decoder.' + name])    

    if args.classifier_path:
        print("Loading pre-trained classifier...")
        checkpoint = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
        for name, param in model.classifier.state_dict().items():
            model.classifier.state_dict()[name].copy_(checkpoint['classifier.' + name])    


    """ Initialize Discriminator """
    discriminator = Discriminator(use_cuda=use_cuda,
                                  output_dim=1)
    if args.discriminator_path:
        print("loading pretrained discriminator...")
        checkpoint = torch.load(args.discriminator_path, map_location=lambda storage, loc: storage)
        for name, param in discriminator.state_dict().items():
            discriminator.state_dict()[name].copy_(checkpoint['discriminator.' + name])


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


    """ Initialize Optimizer """
    '''
    model_opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.learning_rate, 
                           betas=(0.9, 0.99))
    '''
    
    model_opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate, 
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    model_opt.zero_grad()

    discriminator_opt = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), 
                                   lr=args.learning_rate, 
                                   betas=(0.9, 0.99))
    discriminator_opt.zero_grad()

    source_label = 0
    target_label = 1
    
    """ Initialize writer """
    writer = SummaryWriter()

    
    """ Train Model and Fix Discriminator """
    for param in model.extractor.parameters():
        param.requires_grad = False

    """ Train Model and Fix Discriminator """
    for param in model.classifier.parameters():
        param.requires_grad = False

    best_rank1 = 0

    """ Starts Training """
    for step in range(args.num_steps):
    
        model.train()
        discriminator.train()

        cls_loss_value = 0
        rec_loss_value = 0
        contra_loss_value = 0
        adv_target_loss_value = 0
        dis_loss_value = 0 # Discriminator's loss

        model_opt.zero_grad()
        #adjust_model_lr(model_opt, step)

        discriminator_opt.zero_grad()
        #adjust_discriminator_lr(discriminator_opt, step)

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

            image = batch['image'].cuda(args.gpu)
            label = batch['label']
            rec_image = batch['rec_image'].cuda(args.gpu)

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
                rec_loss = loss_rec(pred=rec_source, gt=rec_image, use_cuda=use_cuda)
                loss += args.w_rec * rec_loss
                rec_loss_value += rec_loss.data.cpu().numpy() / args.iter_size / 2

            loss = loss / args.iter_size
            loss.backward()
            
 
            ''' write image '''
            for i in range(len(rec_source)):
                rec_source[i] = inv_normalize(rec_source[i])
                image[i] = inv_normalize(image[i])

            writer.add_image('source image', make_grid(image, nrow=8), step+1)
            writer.add_image('source reconstructed image', make_grid(rec_source, nrow=8), step+1)


            """ Train Target Data """
            _, batch = target_iter.next()
            try:
                _, batch = target_iter.next()
            except:
                target_iter = enumerate(target_loader)
                _, batch = target_iter.next()

            image = batch['image'].cuda(args.gpu)
            rec_image = batch['rec_image'].cuda(args.gpu)


            latent_target, extracted_target, cls_target, rec_target = model(image)

            D_output = discriminator(extracted_target)

            tensor = Variable(torch.FloatTensor(D_output.data.size()).fill_(source_label)).cuda(args.gpu)

            loss = 0

            if args.rec_loss:
                rec_loss = loss_rec(pred=rec_target, gt=rec_image, use_cuda=use_cuda)
                loss += args.w_rec * rec_loss
                rec_loss_value += rec_loss.data.cpu().numpy() / args.iter_size / 2

            if args.adv_loss:
                adv_loss = loss_adv(pred=D_output, gt=tensor)
                adv_target_loss_value += adv_loss.data.cpu().numpy() / args.iter_size

                loss += args.w_adv * adv_loss

            loss = loss / args.iter_size
            loss.backward()
            

            ''' write image '''
            for i in range(len(rec_target)):
                rec_target[i] = inv_normalize(rec_target[i])
                image[i] = inv_normalize(image[i])

            writer.add_image('target image', make_grid(image, nrow=8), step+1)
            writer.add_image('target reconstructed image', make_grid(rec_target, nrow=8), step+1)


            """ Train Discriminator """
            for param in discriminator.parameters():
                param.requires_grad = True

            """ Train with Source Data """
            extracted_source = extracted_source.detach()
            
            D_output = discriminator(extracted_source)

            tensor = Variable(torch.FloatTensor(D_output.data.size()).fill_(source_label)).cuda(args.gpu)
            if args.dis_loss:
                dis_loss = loss_adv(pred=D_output, gt=tensor) / args.iter_size / 2
                dis_loss_value += dis_loss.data.cpu().numpy()

                loss = args.w_dis * dis_loss
                loss.backward()


            """ Train with Target Data """
            extracted_target = extracted_target.detach()
 
            D_output = discriminator(extracted_target)

            tensor = Variable(torch.FloatTensor(D_output.data.size()).fill_(target_label)).cuda(args.gpu)
            if args.dis_loss:
                dis_loss = loss_adv(pred=D_output, gt=tensor) / args.iter_size / 2
                dis_loss_value += dis_loss.data.cpu().numpy()

                loss = args.w_dis * dis_loss
                loss.backward()

        model_opt.step()
        discriminator_opt.step()

        if (step+1) % 1000 == 0:
            print('Start evaluation...')
            model.eval()
            rank1 = eval_metric(args, model, test_loader, query_loader)
            
            writer.add_scalar('rank_1', rank1, (step+1)/1000)

            if rank1 >= best_rank1:
                best_rank1 = rank1
                save_model(model, discriminator, step+1)


        print('[{0:6d}/{1:6d}] cls: {2:.6f} rec: {3:.3f} contra: {4:.3f} adv: {5:.3f} dis: {6:.3f}'.format(step+1, 
            args.num_steps, cls_loss_value, rec_loss_value, contra_loss_value, adv_target_loss_value, dis_loss_value))
        
        """ Write scalar """
        writer.add_scalar('class_loss', cls_loss_value, step+1)
        writer.add_scalar('recon_loss', rec_loss_value, step+1)
        writer.add_scalar('ctrs_loss', contra_loss_value, step+1)
        writer.add_scalar('advG_loss', adv_target_loss_value, step+1)
        writer.add_scalar('advD_loss', dis_loss_value, step+1)
        

        if step >= args.num_steps_stop - 1:
            print('Saving model...')
            save_model(model, discriminator, args.num_steps)

        if (step+1) % args.save_steps == 0:
            print('Saving model...')
            save_model(model, discriminator, step+1)

if __name__ == '__main__':
    main()
