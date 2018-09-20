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
from util.util import save_model, init_model
from util.util import init_resolution_D, init_ACGAN
from util.util import init_source_data, init_target_data
from util.util import init_test_data, init_query_data
from util.util import init_model_optim, init_D_optim
from util.util import inv_normalize, calc_gradient_penalty, make_one_hot
#from model.network import AdaptReID
from model.network import AdaptVAEReID
from model.discriminator import Discriminator, ACGAN
from loss.loss import ClassificationLoss, ReconstructionLoss
from loss.loss import TripletLoss, GlobalLoss, LocalLoss
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


def loss_diff(f1,f2):
    
    f1 = f1.view(f1.size()[0],-1)
    f2 = f2.view(f2.size()[0],-1)
    cosine_diff = F.cosine_similarity(f1,f2).mean()
    
    return cosine_diff


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

    D_resolution = init_resolution_D(args)

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

    D_resolution_opt = init_D_optim(args, D_resolution)

    D_ACGAN_opt = init_D_optim(args, D_ACGAN)


    """ Initialize Writer """
    writer = SummaryWriter()

    best_rank1 = 0
    

    """ Start Training """
    for step in range(args.num_steps):
 
        model.train()
        D_resolution.train()
        D_ACGAN.train()
        
        diff_loss_value = 0
        
        cls_loss_value = 0
        rec_loss_value = 0
        
        KL_loss_value = 0
        GP_loss_value = 0
 
        global_loss_value = 0
        local_loss_value = 0
 
        D_resolution_adv_loss_value = 0
        D_resolution_dis_loss_value = 0

        D_ACGAN_adv_loss_value = 0
        D_ACGAN_dis_loss_value = 0
        D_ACGAN_cls_loss_value = 0

        model_opt.zero_grad()
        D_resolution_opt.zero_grad()
        D_ACGAN_opt.zero_grad()


        """ Train Model and Fix Discriminators """
        for param in D_resolution.parameters():
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


        """ Model Return """
        source_dict = model(image, insert_attrs=make_one_hot(label, args))


        """ Source Training Loss """
        loss = 0

        if args.KL_loss:
            kl_loss = loss_KL(mu=source_dict['mu'], 
                              logvar=source_dict['logvar'])

            KL_loss_value += kl_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_KL * kl_loss


        if args.cls_loss:
            cls_loss = loss_cls(pred=source_dict['cls_vector'], 
                                gt=label, 
                                use_cuda=use_cuda)

            cls_loss_value += cls_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_cls * cls_loss


        if args.triplet_loss:
            global_loss, local_loss = loss_triplet(global_feature=source_dict['global_feature'],
                                                   local_feature=source_dict['local_feature'],
                                                   label=label)

            global_loss_value += global_loss.data.cpu().numpy() / args.iter_size / 2.0
            local_loss_value += local_loss.data.cpu().numpy() / args.iter_size / 2.0

            loss += args.w_global * global_loss
            loss += args.w_local * local_loss


        if args.rec_loss:
            rec_loss = loss_rec(pred=source_dict['rec_image'], 
                                gt=rec_image, 
                                use_cuda=use_cuda)

            rec_loss_value += rec_loss.data.cpu().numpy() / args.iter_size
            loss += args.w_rec * rec_loss
        
        # Difference loss
        if args.diff_loss:
            for f1, f2 in zip(source_dict['skip_e'],source_dict['features']):
                diff_loss = loss_diff(f1, f2)
                diff_loss_value += diff_loss.data.cpu().numpy() / 2.0 / len(source_dict['skip_e'])
            loss += args.w_diff * diff_loss
        
        loss = loss / args.iter_size
        loss.backward()


        """ Visualize Reconstructed Source Image """
        if (step+1) % args.image_steps == 0:
            for i in range(len(source_dict['rec_image'])):
                source_dict['rec_image'][i] = inv_normalize(source_dict['rec_image'][i])
                image[i] = inv_normalize(image[i])

            writer.add_image('HR image', make_grid(image, nrow=16), step+1)
            writer.add_image('Reconstructed HR image', make_grid(source_dict['rec_image'], nrow=16), step+1)


        """ Train Target Data """
        try:
            _, batch = next(target_iter)
        except:
            target_iter = enumerate(target_loader)
            _, batch = next(target_iter)

        image = batch['image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        label = batch['label'].view(-1)
        rec_image = batch['rec_image'].cuda(args.gpu).view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)


        """ Model Return """
        target_dict = model(image, insert_attrs=make_one_hot(label, args))


        """ Resolution Discriminator """
        D_resolution_output = D_resolution(target_dict['resolution_feature'])
        D_resolution_tensor = Variable(torch.FloatTensor(D_resolution_output.data.size()).fill_(config.HR_label)).cuda(args.gpu)


        """ ACGAN Discriminator """
        D_ACGAN_output, cls_ACGAN = D_ACGAN(target_dict['rec_image'])
        D_ACGAN_tensor = Variable(torch.FloatTensor(D_ACGAN_output.data.size()).fill_(config.REAL)).cuda(args.gpu)


        """ Target Training Loss """
        loss = 0

        if args.KL_loss:
            kl_loss = loss_KL(mu=target_dict['mu'], 
                              logvar=target_dict['logvar'])

            KL_loss_value += kl_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_KL * kl_loss
        
        
        #if args.rec_loss:
        #    rec_loss = loss_rec(pred=target_dict['rec_image'], 
        #                        gt=rec_image, 
        #                        use_cuda=use_cuda)

<<<<<<< HEAD
        #    rec_loss_value += rec_loss.data.cpu().numpy() / args.iter_size /2.0
        #    loss += args.w_rec * rec_loss

=======
>>>>>>> a570748f708217682dbade01c4ccac15332ad449
        if args.cls_loss:
            cls_loss = loss_cls(pred=target_dict['cls_vector'], 
                                gt=label, 
                                use_cuda=use_cuda)

            cls_loss_value += cls_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_cls * cls_loss

        if args.triplet_loss:
            global_loss, local_loss = loss_triplet(global_feature=target_dict['global_feature'],
                                                   local_feature=target_dict['local_feature'],
                                                   label=label)

            global_loss_value += global_loss.data.cpu().numpy() / args.iter_size / 2.0
            local_loss_value += local_loss.data.cpu().numpy() / args.iter_size / 2.0

            loss += args.w_global * global_loss
            loss += args.w_local * local_loss


        if args.adv_loss:

            D_resolution_adv_loss = loss_adv(pred=D_resolution_output, 
                                             gt=D_resolution_tensor)

            D_resolution_adv_loss_value += D_resolution_adv_loss.data.cpu().numpy() / args.iter_size
            loss += args.w_adv * D_resolution_adv_loss


        if args.acgan_cls_loss:
            D_ACGAN_cls_loss = loss_cls(pred=cls_ACGAN, 
                                        gt=label, 
                                        use_cuda=use_cuda)

            D_ACGAN_cls_loss_value += D_ACGAN_cls_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_acgan_cls * D_ACGAN_cls_loss


        if args.acgan_adv_loss: 
            D_ACGAN_adv_loss = loss_adv(pred=D_ACGAN_output, 
                                        gt=D_ACGAN_tensor)

            D_ACGAN_adv_loss_value += D_ACGAN_adv_loss.data.cpu().numpy() / args.iter_size
            loss += args.w_acgan_adv * D_ACGAN_adv_loss
            
            
        # Difference loss
        if args.diff_loss:
            for f1, f2 in zip(target_dict['skip_e'],target_dict['features']):
                diff_loss = loss_diff(f1, f2)
                diff_loss_value += diff_loss.data.cpu().numpy() / args.iter_size / 2.0 / len(target_dict['skip_e'])
            loss += args.w_diff * diff_loss

        loss = loss / args.iter_size
        loss.backward()
        
        model_opt.step()

        
        """ Train Resolution Discriminator """
        for param in D_resolution.parameters():
            param.requires_grad = True

        D_resolution_opt.zero_grad()


        """ Train with Source Data """
        D_resolution_output = D_resolution(source_dict['resolution_feature'].detach())
        D_resolution_tensor = Variable(torch.FloatTensor(D_resolution_output.data.size()).fill_(config.HR_label)).cuda(args.gpu)

        loss = 0

        loss = 0
        if args.dis_loss:
            D_resolution_dis_loss = loss_adv(pred=D_resolution_output, 
                                             gt=D_resolution_tensor)

            D_resolution_dis_loss_value += D_resolution_dis_loss.data.cpu().numpy() / args.iter_size/ 2.0
            loss += args.w_dis * D_resolution_dis_loss

        
        """ Train with Target Data """
        D_resolution_output = D_resolution(target_dict['resolution_feature'].detach())
        D_resolution_tensor = Variable(torch.FloatTensor(D_resolution_output.data.size()).fill_(config.LR_label)).cuda(args.gpu)

        if args.dis_loss:
            D_resolution_dis_loss = loss_adv(pred=D_resolution_output, 
                                             gt=D_resolution_tensor)

            D_resolution_dis_loss_value += D_resolution_dis_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_dis * D_resolution_dis_loss
            
        loss = loss / args.iter_size    
        loss.backward()

        D_resolution_opt.step()
        
        
        """ Train ACGAN Discriminator """
        for param in D_ACGAN.parameters():
            param.requires_grad = True
            
        D_ACGAN_opt.zero_grad()
        loss = 0

        """ For Fake (Generated) Image """
        D_ACGAN_output, cls_ACGAN = D_ACGAN(target_dict['rec_image'].detach())
        D_ACGAN_tensor = Variable(torch.FloatTensor(D_ACGAN_output.data.size()).fill_(config.FAKE)).cuda(args.gpu)

#         if args.acgan_cls_loss: 
#             D_ACGAN_cls_loss = loss_cls(pred=cls_ACGAN, 
#                                         gt=label, 
#                                         use_cuda=use_cuda)

#             D_ACGAN_cls_loss_value += D_ACGAN_cls_loss.data.cpu().numpy() / args.iter_size / 2.0
#             loss += args.w_acgan_cls * D_ACGAN_cls_loss


        if args.acgan_dis_loss:
            D_ACGAN_dis_loss = loss_adv(pred=D_ACGAN_output, 
                                        gt=D_ACGAN_tensor)

            D_ACGAN_dis_loss_value += D_ACGAN_dis_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_acgan_dis * D_ACGAN_dis_loss
        

        """ For Real Image """
        D_ACGAN_output, cls_ACGAN = D_ACGAN(rec_image)
        D_ACGAN_tensor = Variable(torch.FloatTensor(D_ACGAN_output.data.size()).fill_(config.REAL)).cuda(args.gpu)

        if args.acgan_cls_loss: 
            D_ACGAN_cls_loss = loss_cls(pred=cls_ACGAN, 
                                        gt=label, 
                                        use_cuda=use_cuda)

            D_ACGAN_cls_loss_value += D_ACGAN_cls_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_acgan_cls * D_ACGAN_cls_loss


        if args.acgan_dis_loss:
            D_ACGAN_dis_loss = loss_adv(pred=D_ACGAN_output, 
                                        gt=D_ACGAN_tensor)

            D_ACGAN_dis_loss_value += D_ACGAN_dis_loss.data.cpu().numpy() / args.iter_size / 2.0
            loss += args.w_acgan_dis * D_ACGAN_dis_loss
            

        if args.gp_loss:
            D_ACGAN_gp_loss = calc_gradient_penalty(D_ACGAN, 
                                                    rec_image, 
                                                    target_dict['rec_image'])
            GP_loss_value = D_ACGAN_gp_loss.data.cpu().numpy() / args.iter_size
            loss += D_ACGAN_gp_loss * args.w_gp
            
        loss = loss / args.iter_size    
        loss.backward()

        D_ACGAN_opt.step()
        
        
        """ Visualize Generated Target Image """
        if (step+1) % args.image_steps == 0:
            save_model(args, model, D_resolution, D_ACGAN=D_ACGAN) # To save model
            
            for i in range(len(target_dict['rec_image'])):
                target_dict['rec_image'][i] = inv_normalize(target_dict['rec_image'][i])
                image[i] = inv_normalize(image[i])

            writer.add_image('LR image', make_grid(image, nrow=16), step+1)
            writer.add_image('Generated LR image', make_grid(target_dict['rec_image'], nrow=16), step+1)


        print_string = '[{:6d}/{:6d}]'.format(step+1, args.num_steps)
        
        if args.cls_loss:
            print_string += ' cls: {:.6f}'.format(cls_loss_value)

        if args.rec_loss:
            print_string += ' rec: {:.6f}'.format(rec_loss_value)

        if args.triplet_loss:
            print_string += ' global: {:.6f} local: {:.6f}'.format(global_loss_value, local_loss_value)

        if args.adv_loss:
            print_string += ' adv: {:.6f}'.format(D_resolution_adv_loss_value)

        if args.dis_loss:
            print_string += ' dis: {:.6f}'.format(D_resolution_dis_loss_value)

        if args.acgan_adv_loss:
            print_string += ' AC adv: {:.6f}'.format(D_ACGAN_adv_loss_value)

        if args.acgan_dis_loss:
            print_string += ' AC dis: {:.6f}'.format(D_ACGAN_dis_loss_value)

        if args.acgan_cls_loss:
            print_string += ' AC cls: {:.6f}'.format(D_ACGAN_cls_loss_value)

        if args.KL_loss:
            print_string += ' KL: {:.6f}'.format(KL_loss_value)

        if args.gp_loss:
            print_string += ' GP: {:.6f}'.format(GP_loss_value)
        
        if True:
            print_string += ' Diff: {:.6f}'.format(diff_loss_value)


        print(print_string)

        '''
        print('[{0:6d}/{1:6d}] cls: {2:.6f} rec: {3:.6f} global: {4:.6f} local: {5:.6f} adv: {6:.6f} dis: {7:.6f} AC adv: {8:6f} AC dis: {9:6f} AC cls: {10:6f} KLD: {11:6f} GP: {12:6f}'.format(step+1, 
              args.num_steps, 
              cls_loss_value, 
              rec_loss_value, 
              global_loss_value, 
              local_loss_value,
              D_resolution_adv_loss_value, 
              D_resolution_dis_loss_value,
              D_ACGAN_adv_loss_value, 
              D_ACGAN_dis_loss_value,
              D_ACGAN_cls_loss_value,
              KL_loss_value,
              GP_loss_value))
        '''

        
        """ Write Scalar """
        writer.add_scalar('Classification Loss', cls_loss_value, step+1)
        writer.add_scalar('Reconstruction Loss', rec_loss_value, step+1)
        writer.add_scalar('Adversarial Loss', D_resolution_adv_loss_value, step+1)
        writer.add_scalar('Discriminator Loss', D_resolution_dis_loss_value, step+1)
        writer.add_scalar('ACGAN Adversarial Loss', D_ACGAN_adv_loss_value, step+1)
        writer.add_scalar('ACGAN Discriminator Loss', D_ACGAN_dis_loss_value, step+1)
        writer.add_scalar('ACGAN Classification Loss', D_ACGAN_cls_loss_value, step+1)
        writer.add_scalar('Global Triplet Loss', global_loss_value, step+1)
        writer.add_scalar('Local Triplet Loss', local_loss_value, step+1)
        writer.add_scalar('KLD Loss', KL_loss_value, step+1)
        writer.add_scalar('GP Loss', GP_loss_value, step+1)
        writer.add_scalar('Diff Loss', diff_loss_value, step+1)

        
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
                save_model(args, model, D_resolution, D_ACGAN=D_ACGAN) # To save model
                writer.add_scalar('Best Rank 1', best_rank1, (step+1)/args.eval_steps)

            print('Rank:', rank1, rank5, rank10, rank20)
            print('mAP:', mAP)
            print('Best rank1:', best_rank1)


if __name__ == '__main__':
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    main()
