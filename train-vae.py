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
from model.network import AdaptVAEReID
from model.discriminator import Discriminator
from model.discriminator import ACGAN
from loss.loss import ClassificationLoss
from loss.loss import ReconstructionLoss
from loss.loss import TripletLoss
from loss.loss import GlobalLoss
from loss.loss import LocalLoss
from data.duke import Duke
from data.market import Market
from data.msmt import MSMT
from data.cuhk import CUHK
from data.viper import VIPER
from data.caviar import CAVIAR
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

def loss_KL(mu, logvar):  # NEW added
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return KLD 


def loss_rec(pred, gt, use_cuda=True):
    criterion = ReconstructionLoss(dist_metric='L1', 
                                   use_cuda=use_cuda)
    loss = criterion(pred, gt)
    return loss


def loss_triplet(global_feature, local_feature, label, normalize=True):
    criterion = TripletLoss(margin=config.GLOBAL_MARGIN)
    global_loss, pos_inds, neg_inds = GlobalLoss(criterion, 
                                                  global_feature,
                                                  label.cuda(args.gpu),
                                                  normalize_feature=normalize)

    criterion = TripletLoss(margin=config.LOCAL_MARGIN)
    local_loss = LocalLoss(criterion, 
                            local_feature,
                            pos_inds,
                            neg_inds,
                            label.cuda(args.gpu),
                            normalize_feature=normalize)

    return global_loss, local_loss


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


def save_model(model, D_1=None, D_2=None, D_ACGAN=None):
    extractor_path = os.path.join(args.model_dir, 'Extractor_{}.pth.tar'.format(args.source_dataset))
    decoder_path = os.path.join(args.model_dir, 'Decoder_{}.pth.tar'.format(args.source_dataset))
    classifier_path = os.path.join(args.model_dir, 'Classifier_{}.pth.tar'.format(args.source_dataset))
    D1_path = os.path.join(args.model_dir, 'D1_{}.pth.tar'.format(args.source_dataset))
            
    torch.save(model.extractor.state_dict(), extractor_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    torch.save(model.classifier.state_dict(), classifier_path)
    torch.save(D_1.state_dict(), D1_path)
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
    elif args.source_dataset == 'VIPER':
        classifier_output_dim = config.VIPER_CLASS_NUM
    elif args.source_dataset == 'CAVIAR':
        classifier_output_dim = config.CAVIAR_CLASS_NUM
    
    model = AdaptVAEReID(backbone='resnet-50',
                      use_cuda=use_cuda,
                      classifier_output_dim=classifier_output_dim, code_dim=classifier_output_dim)

    if args.extractor_path:
        print("Loading pre-trained extractor...")
        checkpoint = torch.load(args.extractor_path, map_location=lambda storage, loc: storage)
        for name, param in model.extractor.state_dict().items():
            model.extractor.state_dict()[name].copy_(checkpoint[name])    

    if args.decoder_path:
        print("Loading pre-trained decoder...")
        checkpoint = torch.load(args.decoder_path, map_location=lambda storage, loc: storage)
        for name, param in model.decoder.state_dict().items():
            model.decoder.state_dict()[name].copy_(checkpoint[name])    

    if args.classifier_path:
        print("Loading pre-trained classifier...")
        checkpoint = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
        for name, param in model.classifier.state_dict().items():
            model.classifier.state_dict()[name].copy_(checkpoint[name])    


    """ Initialize Discriminator """
    D_1 = Discriminator(input_channel=config.D1_INPUT_CHANNEL,
                        fc_input_dim=config.D1_FC_INPUT_DIM,
                        use_cuda=use_cuda)

    if args.discriminator_path:
        print("loading pre-trained discriminator...")
        checkpoint = torch.load(args.discriminator_path, map_location=lambda storage, loc: storage)
        for name, param in D_1.state_dict().items():
            D_1.state_dict()[name].copy_(checkpoint[name])


    """ Initialize ACGAN Discriminator """
    D_ACGAN = ACGAN(input_channel=config.D1_INPUT_CHANNEL,
                    fc_input_dim=config.D1_FC_INPUT_DIM,
                    class_num=classifier_output_dim,
                    use_cuda=use_cuda)

    if args.acgan_path:
        print("loading pre-trained ACGAN discriminator...")
        checkpoint = torch.load(args.acgan_path, map_location=lambda storage, loc: storage)
        for name, param in D_ACGAN.state_dict().items():
            D_ACGAN.state_dict()[name].copy_(checkpoint[name])

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
    elif args.source_dataset == 'VIPER':
        SourceData = VIPER
    elif args.source_dataset == 'CAVIAR':
        SourceData = CAVIAR
        
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
    elif args.target_dataset == 'VIPER':
        TargetData = VIPER
        TestData = VIPER
        QueryData = VIPER
    elif args.target_dataset == 'CAVIAR':
        TargetData = CAVIAR
        TestData = CAVIAR
        QueryData = CAVIAR

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

    D1_opt = optim.Adam(filter(lambda p: p.requires_grad, D_1.parameters()), 
                        lr=args.learning_rate / 10.0, 
                        betas=(0.9, 0.99))
    D1_opt.zero_grad()

    D_ACGAN_opt = optim.Adam(filter(lambda p: p.requires_grad, D_ACGAN.parameters()), 
                             lr=args.learning_rate / 10.0, 
                             betas=(0.9, 0.99))
    D_ACGAN_opt.zero_grad()

    HR_label = 0
    LR_label = 1


    """ Initialize writer """
    writer = SummaryWriter()

    best_rank1 = 0
    
    """Test"""
   


    """ Starts Training """
    for step in range(args.num_steps):
 
        model.train()
        D_1.train()
        D_ACGAN.train()

        cls_loss_value = 0
        rec_loss_value = 0
        
        KL_loss_value = 0 # NEW added
 
        global_loss_value = 0
        local_loss_value = 0
 
        D1_adv_loss_value = 0
        D1_dis_loss_value = 0 # Discriminator's loss

        D_ACGAN_adv_loss_value = 0
        D_ACGAN_dis_loss_value = 0 # ACGAN Discriminator's loss
        D_ACGAN_cls_loss_value = 0 # ACGAN classification loss

        model_opt.zero_grad()
        D1_opt.zero_grad()
        D_ACGAN_opt.zero_grad()

        for idx in range(args.iter_size):

            """ Train Model and Fix Discriminators """
            for param in D_1.parameters():
                param.requires_grad = False

            for param in D_ACGAN.parameters():
                param.requires_grad = False


            """ Train Source Data """
            try:
                _, batch = next(source_iter)
            except:
                source_iter = enumerate(source_loader)
                _, batch = next(source_iter)

            image = batch['image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
            label = batch['label'].view(-1)
            rec_image = batch['rec_image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)


            latent_source, features_source, cls_source, rec_source, global_feature_source, local_feature_source, source_mu, source_logvar = model(image)

            extracted_source_low = features_source[-1]

            loss = 0
            
            if False:
                kl_loss = loss_KL(mu=source_mu, logvar=source_logvar)
                KL_loss_value += kl_loss.data.cpu().numpy() / args.iter_size / 2.0
                loss += kl_loss*1e-6
                

            if args.cls_loss:
                """
                    Resolution-Invariant Feature Classification Loss

                    Args:
                        cls_source: batch x class_num (prediction results)
                """
                cls_loss = loss_cls(pred=cls_source, gt=label, use_cuda=use_cuda)
                cls_loss_value += cls_loss.data.cpu().numpy() / args.iter_size / 2.0
                loss += args.w_cls * cls_loss
 
 
            if args.triplet_loss:
                """ 
                    Triplet Loss on Average Pooled Feature Vector
                """
                global_loss, local_loss = loss_triplet(global_feature=global_feature_source,
                                                       local_feature=local_feature_source,
                                                       label=label)

                global_loss_value += global_loss.data.cpu().numpy() / args.iter_size / 2.0
                local_loss_value += local_loss.data.cpu().numpy() / args.iter_size / 2.0

                loss += args.w_global * global_loss
                loss += args.w_local * local_loss


            if args.rec_loss:
                """
                    Image Reconstruction Loss on HR images
                """
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


            latent_target, features_target, cls_target, rec_target, global_feature_target, local_feature_target, target_mu, target_logvar = model(image)

            extracted_target_low = features_target[-1] # f_5

            D1_output = D_1(extracted_target_low)
            D1_tensor = Variable(torch.FloatTensor(D1_output.data.size()).fill_(HR_label)).cuda(args.gpu)

            D_ACGAN_output, cls_ACGAN = D_ACGAN(extracted_target_low)
            D_ACGAN_tensor = Variable(torch.FloatTensor(D_ACGAN_output.data.size()).fill_(HR_label)).cuda(args.gpu)

            loss = 0

            #if args.rec_loss:
            #    rec_loss = loss_rec(pred=rec_target, gt=rec_image, use_cuda=use_cuda)
            #    rec_loss_value += rec_loss.data.cpu().numpy() / args.iter_size / 2.0
            #    loss += args.w_rec * rec_loss
            
            if False:
                kl_loss = loss_KL(mu=target_mu, logvar=target_logvar)
                KL_loss_value += kl_loss.data.cpu().numpy() / args.iter_size / 2.0
                loss += kl_loss*1e-6

            if args.cls_loss:
                """
                    Resolution-Invariant Feature Classification Loss

                    Args:
                        cls_source: batch x class_num (prediction results)
                """
                cls_loss = loss_cls(pred=cls_target, gt=label, use_cuda=use_cuda)
                cls_loss_value += cls_loss.data.cpu().numpy() / args.iter_size / 2.0
                loss += args.w_cls * cls_loss

                """
                    ACGAN Classification Loss

                    Args:
                        cls_ACGAN: batch
                """
                D_ACGAN_cls_loss = loss_cls(pred=cls_ACGAN, gt=label, use_cuda=use_cuda)
                D_ACGAN_cls_loss_value += D_ACGAN_cls_loss.data.cpu().numpy() / args.iter_size / 2.0
                loss += args.w_cls * D_ACGAN_cls_loss
 
                
            if args.triplet_loss:
                """ 
                    Triplet Loss on Average Pooled Feature Vector
                """
                global_loss, local_loss = loss_triplet(global_feature=global_feature_target,
                                                       local_feature=local_feature_target,
                                                       label=label)

                global_loss_value += global_loss.data.cpu().numpy() / args.iter_size / 2.0
                local_loss_value += local_loss.data.cpu().numpy() / args.iter_size / 2.0

                loss += args.w_global * global_loss
                loss += args.w_local * local_loss


            if args.adv_loss:
                """
                    Adversarial Loss on Resolution-Invariant Feature

                    Args:
                        D1_output: batch x 1 (HR or LR)
                """
                D1_adv_loss = loss_adv(pred=D1_output, gt=D1_tensor)
                D1_adv_loss_value += D1_adv_loss.data.cpu().numpy() / args.iter_size
                loss += args.w_adv * D1_adv_loss

                """
                    Adversarial Loss (PatchGAN) on Reconstructed HR images (LR -> HR)

                    Args:
                        D_ACGAN_output: b x c x h x w (HR or LR)
                """
                D_ACGAN_adv_loss = loss_adv(pred=D_ACGAN_output, gt=D_ACGAN_tensor)
                D_ACGAN_adv_loss_value += D_ACGAN_adv_loss.data.cpu().numpy() / args.iter_size
                loss += args.w_adv * D_ACGAN_adv_loss

            loss = loss / args.iter_size
            loss.backward()
            

            """ Write Images """
            if (step+1) % args.image_steps == 0:
                for i in range(len(rec_target)):
                    rec_target[i] = inv_normalize(rec_target[i])
                    image[i] = inv_normalize(image[i])

                writer.add_image('LR image', make_grid(image, nrow=16), step+1)
                writer.add_image('Reconstructed LR image', make_grid(rec_target, nrow=16), step+1)


            """ Train Feature-Level Discriminator """
            for param in D_1.parameters():
                param.requires_grad = True


            """ Train with Source Data """
            extracted_source_low = extracted_source_low.detach()
            
            D1_output = D_1(extracted_source_low)
            D1_tensor = Variable(torch.FloatTensor(D1_output.data.size()).fill_(HR_label)).cuda(args.gpu)

            if args.dis_loss:
                """
                    Discriminator Loss on Resolution-Invariant Feature

                    Args:
                        D1_output: batch x 1 (HR or LR)
                """
                D1_dis_loss = loss_adv(pred=D1_output, gt=D1_tensor) / args.iter_size / 2.0
                D1_dis_loss_value += D1_dis_loss.data.cpu().numpy()

                loss = args.w_dis * D1_dis_loss
                loss.backward()


            """ Train with Target Data """
            extracted_target_low = extracted_target_low.detach()
 
            D1_output = D_1(extracted_target_low)
            D1_tensor = Variable(torch.FloatTensor(D1_output.data.size()).fill_(LR_label)).cuda(args.gpu)

            if args.dis_loss:
                """
                    Discriminator Loss on Resolution-Invariant Feature

                    Args:
                        D1_output: batch x 1 (HR or LR)
                """
                D1_dis_loss = loss_adv(pred=D1_output, gt=D1_tensor) / args.iter_size / 2.0
                D1_dis_loss_value += D1_dis_loss.data.cpu().numpy()

                loss = args.w_dis * D1_dis_loss
                loss.backward()


            """ Train Image-Level Discriminator """
            for param in D_ACGAN.parameters():
                param.requires_grad = True

            D_ACGAN_output, cls_ACGAN = D_ACGAN(extracted_target_low)
            D_ACGAN_tensor = Variable(torch.FloatTensor(D_ACGAN_output.data.size()).fill_(LR_label)).cuda(args.gpu)

            if args.dis_loss:
                """
                    Adversarial Loss (PatchGAN) on Reconstructed HR images (LR -> HR)

                    Args:
                        D_ACGAN_output: b x c x h x w (HR or LR)
                """
                D_ACGAN_dis_loss = loss_adv(pred=D_ACGAN_output, gt=D_ACGAN_tensor)
                D_ACGAN_dis_loss_value += D_ACGAN_dis_loss.data.cpu().numpy() / args.iter_size
                loss += args.w_dis * D_ACGAN_dis_loss

        model_opt.step()
        D1_opt.step()
        D_ACGAN_opt.step()


        print('[{0:6d}/{1:6d}] cls: {2:.6f} rec: {3:.6f} global: {4:.6f} local: {5:.6f} adv: {6:.6f} dis: {7:.6f}, AC adv: {8:6f}, AC dis: {9:6f}, AC cls: {10:6f}, KLD: {11:6f}'.format(step+1, 
              args.num_steps, 
              cls_loss_value, 
              rec_loss_value, 
              global_loss_value, 
              local_loss_value,
              D1_adv_loss_value, 
              D1_dis_loss_value,
              D_ACGAN_adv_loss_value, 
              D_ACGAN_dis_loss_value,
              D_ACGAN_cls_loss_value,
              KL_loss_value))
        
        """ Write Scalar """
        writer.add_scalar('Classification Loss', cls_loss_value, step+1)
        writer.add_scalar('Reconstruction Loss', rec_loss_value, step+1)
        writer.add_scalar('Adversarial Loss', D1_adv_loss_value, step+1)
        writer.add_scalar('Discriminator Loss', D1_dis_loss_value, step+1)
        writer.add_scalar('ACGAN Adversarial Loss', D_ACGAN_adv_loss_value, step+1)
        writer.add_scalar('ACGAN Discriminator Loss', D_ACGAN_dis_loss_value, step+1)
        writer.add_scalar('ACGAN Classification Loss', D_ACGAN_cls_loss_value, step+1)
        writer.add_scalar('Global Triplet Loss', global_loss_value, step+1)
        writer.add_scalar('Local Triplet Loss', local_loss_value, step+1)
        writer.add_scalar('KLD Loss', KL_loss_value, step+1)

        
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
                save_model(model, D_1)
                writer.add_scalar('Best Rank 1', best_rank1, (step+1)/args.eval_steps)

            print('Rank:', rank1, rank5, rank10, rank20)
            print('mAP:', mAP)
            print('Best rank1:', best_rank1)


if __name__ == '__main__':
    main()
