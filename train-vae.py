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
from util.util import save_model
from util.util import init_model
from util.util import init_resolution_D
from util.util import init_ACGAN
from util.util import init_source_data
from util.util import init_target_data
from util.util import init_test_data
from util.util import init_query_data
from util.util import init_model_optim
from util.util import init_D_optim
from util.util import inv_normalize
from util.util import calc_gradient_penalty
from util.util import make_one_hot
from model.network import AdaptReID
from model.network import AdaptVAEReID
from model.discriminator import Discriminator
from model.discriminator import ACGAN
from loss.loss import ClassificationLoss
from loss.loss import ReconstructionLoss
from loss.loss import TripletLoss
from loss.loss import GlobalLoss
from loss.loss import LocalLoss
from parser.parser import ArgumentParser
from util.eval_utils import eval_metric
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import config



""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='train').parse()


def loss_KL(mu, logvar):
    
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
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


def main():

    """ GPU Settings """
    torch.cuda.set_device(args.gpu)
    use_cuda = torch.cuda.is_available()


    """ Initialize Model and Discriminators """
    model = init_model(args)

    D_1 = init_resolution_D(args)

    D_ACGAN = init_ACGAN(args)


    """ Initialize Data """
    source_data, source_loader = init_source_data(args)
    source_iter = enumerate(source_loader)

    target_data, target_loader = init_target_data(args)
    target_iter = enumerate(target_loader)

    test_data, test_loader = init_test_data(args)

    query_data, query_loader = init_query_data(args)


    """ Initialize Optimizers """
    model_opt = init_model_optim(args, model)

    D1_opt = init_D_optim(args, D_1)

    D_ACGAN_opt = init_D_optim(args, D_ACGAN)


    """ Initialize writer """
    writer = SummaryWriter()

    best_rank1 = 0
    

    """ Starts Training """
    for step in range(args.num_steps):
 
        model.train()
        D_1.train()
        D_ACGAN.train()

        cls_loss_value = 0
        rec_loss_value = 0
        
        KL_loss_value = 0
        GP_loss_value = 0
 
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


        latent_source, features_source, cls_source, rec_source, global_feature_source, local_feature_source, source_mu, source_logvar = model(image, insert_attrs=make_one_hot(label, args))

        extracted_source_low = features_source[-1]

        loss = 0

        if args.KL_loss:
            kl_loss = loss_KL(mu=source_mu, logvar=source_logvar)
            KL_loss_value += kl_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += kl_loss


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


        """ Write Images For Source"""
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


        latent_target, features_target, cls_target, rec_target, global_feature_target, local_feature_target, target_mu, target_logvar = model(image, insert_attrs=make_one_hot(label, args))

        extracted_target_low = features_target[-1] # f_5

        D1_output = D_1(extracted_target_low)
        D1_tensor = Variable(torch.FloatTensor(D1_output.data.size()).fill_(config.HR_label)).cuda(args.gpu)

        D_ACGAN_output, cls_ACGAN = D_ACGAN(rec_target)
        D_ACGAN_tensor = Variable(torch.FloatTensor(D_ACGAN_output.data.size()).fill_(config.HR_label)).cuda(args.gpu)

        loss = 0

        ### To pretrain Decoder
        if args.rec_loss:
            rec_loss = loss_rec(pred=rec_target, gt=rec_image, use_cuda=use_cuda)
            rec_loss_value += rec_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_rec * rec_loss
        ###

        if args.KL_loss:
            kl_loss = loss_KL(mu=target_mu, logvar=target_logvar)
            KL_loss_value += kl_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_KL * kl_loss

        if args.cls_loss:
            """
                Resolution-Invariant Feature Classification Loss

                Args:
                    cls_source: batch x class_num (prediction results)
            """
            cls_loss = loss_cls(pred=cls_target, gt=label, use_cuda=use_cuda)
            cls_loss_value += cls_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_cls * cls_loss

        if args.acgan_cls_loss:    
            """
                ACGAN Classification Loss

                Args:
                    cls_ACGAN: batch
            """
            D_ACGAN_cls_loss = loss_cls(pred=cls_ACGAN, gt=label, use_cuda=use_cuda)
            D_ACGAN_cls_loss_value += D_ACGAN_cls_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_acgan_cls * D_ACGAN_cls_loss
            


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

        if args.acgan_adv_loss:    
            """
                Adversarial Loss (PatchGAN) on Reconstructed HR images (LR -> HR)

                Args:
                    D_ACGAN_output: b x c x h x w (HR or LR)
            """
            D_ACGAN_adv_loss = loss_adv(pred=D_ACGAN_output, gt=D_ACGAN_tensor)
#             D_ACGAN_adv_loss = - D_ACGAN_output.mean() #WGAN
            D_ACGAN_adv_loss_value += D_ACGAN_adv_loss.data.cpu().numpy() / args.iter_size
            loss += args.w_acgan_adv * D_ACGAN_adv_loss

        loss = loss / args.iter_size
        loss.backward()
        
        model_opt.step()

        
        """ Train Feature-Level Discriminator """
        for param in D_1.parameters():
            param.requires_grad = True
        D1_opt.zero_grad()


        """ Train with Source Data """
        extracted_source_low = extracted_source_low.detach()

        D1_output = D_1(extracted_source_low)
        D1_tensor = Variable(torch.FloatTensor(D1_output.data.size()).fill_(config.HR_label)).cuda(args.gpu)

        loss = 0
        if args.dis_loss:
            """
                Discriminator Loss on Resolution-Invariant Feature

                Args:
                    D1_output: batch x 1 (HR or LR)
            """
            D1_dis_loss = loss_adv(pred=D1_output, gt=D1_tensor) / args.iter_size / 2.0
            D1_dis_loss_value += D1_dis_loss.data.cpu().numpy()

            loss += args.w_dis * D1_dis_loss
            

        
        """ Train with Target Data """
        extracted_target_low = extracted_target_low.detach()

        D1_output = D_1(extracted_target_low)
        D1_tensor = Variable(torch.FloatTensor(D1_output.data.size()).fill_(config.LR_label)).cuda(args.gpu)

        if args.dis_loss:
            """
                Discriminator Loss on Resolution-Invariant Feature

                Args:
                    D1_output: batch x 1 (HR or LR)
            """
            D1_dis_loss = loss_adv(pred=D1_output, gt=D1_tensor) / args.iter_size / 2.0
            D1_dis_loss_value += D1_dis_loss.data.cpu().numpy()

            loss += args.w_dis * D1_dis_loss
            
#         loss = loss / args.iter_size    
#         loss.backward()

#         D1_opt.step()
        
        
        """ Train Image-Level Discriminator """
        for param in D_ACGAN.parameters():
            param.requires_grad = True
            
        D_ACGAN_opt.zero_grad()
        loss = 0
        
        # For Fake image
        D_ACGAN_output, _ = D_ACGAN(rec_target.detach())
        D_ACGAN_tensor = Variable(torch.FloatTensor(D_ACGAN_output.data.size()).fill_(config.LR_label)).cuda(args.gpu)

        if args.acgan_adv_loss:
            """
                Adversarial Loss (PatchGAN) on Reconstructed HR images (LR -> HR)

                Args:
                    D_ACGAN_output: b x c x h x w (HR or LR)
            """
            D_ACGAN_dis_loss = loss_adv(pred=D_ACGAN_output, gt=D_ACGAN_tensor)
#             D_ACGAN_dis_loss = D_ACGAN_output.mean() # WGAN
            D_ACGAN_dis_loss_value += D_ACGAN_dis_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_acgan_adv * D_ACGAN_dis_loss
        
        # For Real image
        D_ACGAN_output, cls_ACGAN = D_ACGAN(rec_image)
        D_ACGAN_tensor = Variable(torch.FloatTensor(D_ACGAN_output.data.size()).fill_(config.HR_label)).cuda(args.gpu)

        if args.acgan_adv_loss:
            """
                Adversarial Loss (PatchGAN) on Reconstructed HR images (LR -> HR)

                Args:
                    D_ACGAN_output: b x c x h x w (HR or LR)
            """
            D_ACGAN_dis_loss = loss_adv(pred=D_ACGAN_output, gt=D_ACGAN_tensor)
#             D_ACGAN_dis_loss = - D_ACGAN_output.mean() # WGAN
            D_ACGAN_dis_loss_value += D_ACGAN_dis_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_acgan_adv * D_ACGAN_dis_loss
            
        if args.acgan_cls_loss:    
            
            D_ACGAN_cls_loss = loss_cls(pred=cls_ACGAN, gt=label, use_cuda=use_cuda)
            D_ACGAN_cls_loss_value += D_ACGAN_cls_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_acgan_cls * D_ACGAN_cls_loss
            
            # Gradient Panelty
        if args.gp_loss:
            
            D_ACGAN_gp_loss = calc_gradient_penalty(D_ACGAN,rec_image,rec_target)
            GP_loss_value = D_ACGAN_gp_loss.data.cpu().numpy() / args.iter_size
            loss += D_ACGAN_gp_loss * args.w_gp
            
            

        loss = loss / args.iter_size    
        loss.backward()
        D_ACGAN_opt.step()
        
        
        """ Write Images For target"""
        if (step+1) % args.image_steps == 0:
            save_model(args, model, D_1, D_ACGAN=D_ACGAN) # To save model
            
            for i in range(len(rec_target)):
                rec_target[i] = inv_normalize(rec_target[i])
                image[i] = inv_normalize(image[i])

            writer.add_image('LR image', make_grid(image, nrow=16), step+1)
            writer.add_image('Reconstructed LR image', make_grid(rec_target, nrow=16), step+1)


        print('[{0:6d}/{1:6d}] cls: {2:.6f} rec: {3:.6f} global: {4:.6f} local: {5:.6f} adv: {6:.6f} dis: {7:.6f}, AC adv: {8:6f}, AC dis: {9:6f}, AC cls: {10:6f}, KLD: {11:6f}, GP: {12:6f}'.format(step+1, 
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
              KL_loss_value,
              GP_loss_value))
        
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
        writer.add_scalar('GP Loss', GP_loss_value, step+1)

        
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
                #save_model(args, model, D_1, D_ACGAN=D_ACGAN) # To save model
                writer.add_scalar('Best Rank 1', best_rank1, (step+1)/args.eval_steps)

            print('Rank:', rank1, rank5, rank10, rank20)
            print('mAP:', mAP)
            print('Best rank1:', best_rank1)


if __name__ == '__main__':
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    main()
