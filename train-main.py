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
from model.discriminator import *
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
    # elif args.target_dataset == 'MSMT':
    #     classifier_output_dim = config.MSMT_CLASS_NUM
    # elif args.target_dataset == 'CUHK':
    #     classifier_output_dim = config.CUHK_CLASS_NUM
    # elif args.target_dataset == 'VIPER':
    #     classifier_output_dim = config.VIPER_CLASS_NUM
    # elif args.target_dataset == 'CAVIAR':
    #     classifier_output_dim = config.CAVIAR_CLASS_NUM
    # elif args.target_dataset == 'VERI':
    #     classifier_output_dim = config.VERI_CLASS_NUM
    # elif args.target_dataset == 'VRIC':
    #     classifier_output_dim = config.VRIC_CLASS_NUM    


    model = Sense_ReID(backbone='resnet-50',
                         use_cuda=use_cuda,
                         classifier_output_dim=classifier_output_dim)
    
    
#     model_dict = model.state_dict()

#     # 1. filter out unnecessary keys
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dict) 
#     # 3. load the new state dict
#     model.load_state_dict(pretrained_dict)

    if args.extractor_path:
        print("Loading pre-trained extractor...")
        checkpoint = torch.load(args.extractor_path, map_location=lambda storage, loc: storage)
#         for name, param in model.extractor.state_dict().items():
#             model.extractor.state_dict()[name].copy_(checkpoint[name])
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

    # s_extractor_path = os.path.join(args.model_dir, 'S_Extractor_{}.pth.tar'.format(args.source_dataset))
    c_extractor_path = os.path.join(args.model_dir, 'C_Extractor_{}.pth.tar'.format(args.source_dataset))
    classifier_path = os.path.join(args.model_dir, 'Classifier_{}.pth.tar'.format(args.source_dataset))
    generator_path = os.path.join(args.model_dir, 'Generator_{}.pth.tar'.format(args.source_dataset))
    discriminator_path = os.path.join(args.model_dir, 'Discriminator_{}.pth.tar'.format(args.source_dataset))

    # torch.save(model.structure_extractor.state_dict(), s_extractor_path)
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
    # elif args.target_dataset == 'VRIC':
    #     TargetData = VRIC


    target_data = TargetData(mode='train',
                             transform=NormalizeImage(['image', 'rec_image']),
                             random_crop=args.random_crop,ds_factor=1,im_per_id=1)

    target_loader = DataLoader(target_data,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=True)

    return target_data, target_loader


# def init_test_data(args):

#     if args.target_dataset == 'Duke':
#         TestData = Duke
#     elif args.target_dataset == 'Market':
#         TestData = Market
#     elif args.target_dataset == 'MSMT':
#         TestData = MSMT
#     elif args.target_dataset == 'CUHK':
#         TestData = CUHK
#     elif args.target_dataset == 'VIPER':
#         TestData = VIPER
#     elif args.target_dataset == 'CAVIAR':
#         TestData = CAVIAR
#     elif args.target_dataset == 'VERI':
#         TestData = VERI
#     # elif args.target_dataset == 'VRIC':
#     #     TestData = VRIC

#     test_data = TestData(mode='test',
#                          transform=NormalizeImage(['image']),ds_factor=1,g_gray=True)

#     test_loader = DataLoader(test_data,
#                              batch_size=int(args.batch_size),
#                              num_workers=args.num_workers,
#                              pin_memory=True)

#     return test_data, test_loader


# def init_query_data(args):

#     if args.target_dataset == 'Duke':
#         QueryData = Duke
#     elif args.target_dataset == 'Market':
#         QueryData = Market
#     elif args.target_dataset == 'MSMT':
#         QueryData = MSMT
#     elif args.target_dataset == 'CUHK':
#         QueryData = CUHK
#     elif args.target_dataset == 'VIPER':
#         QueryData = VIPER
#     elif args.target_dataset == 'CAVIAR':
#         QueryData = CAVIAR
#     elif args.target_dataset == 'VERI':
#         QueryData = VERI
#     # elif args.target_dataset == 'VRIC':
#     #     QueryData = VRIC


#     query_data = QueryData(mode='query',
#                            transform=NormalizeImage(['image']),ds_factor=1,q_gray=True)

#     query_loader = DataLoader(query_data,
#                               batch_size=int(args.batch_size),
#                               num_workers=args.num_workers,
#                               pin_memory=True)

#     return query_data, query_loader

""" Parse Arguments """ 
args, arg_groups = ArgumentParser(mode='train').parse()

def loss_triplet(pred, gt, use_cuda=True):
    criterion = TripletLoss(margin=None)
#     criterion = TripletLoss(margin=config.GLOBAL_MARGIN)
#     criterion = ClassificationLoss(use_cuda=use_cuda)
#     loss = criterion(pred, gt)
    loss, _, _ = criterion(pred, gt.cuda())
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

    model = model.module
    
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


    """ Initialize Optimizers """
    
    lr_g = 1e-4
    lr_d = 1e-4
    # Setup the optimizers
    beta1 = 0
    beta2 = 0.999
    weight_decay = 0.0005             
    dis_params = list(dis_model.parameters()) #+ list(self.dis_b.parameters())
    gen_params = list(model.generator.parameters()) #+ list(self.gen_b.parameters())

    # domain_params = list(model.)
    
    dis_b = dis_a = dis_model

    dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                    lr=lr_d, betas=(beta1, beta2), weight_decay=weight_decay)
    
    gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                    lr=lr_g, betas=(beta1, beta2), weight_decay=weight_decay)
    
    instancenorm = nn.InstanceNorm2d(512, affine=False)
    
    
    ignored_params = (
                        list(map(id, model.structure_classifier.parameters()))
                    + list(map(id, model.D_domain.parameters()))
                    +list(map(id, model.generator.parameters())))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    
    lr2 = 2e-3
    model_opt = torch.optim.SGD([
            {'params': base_params, 'lr': lr2},
            {'params': model.structure_classifier.parameters(), 'lr': lr2*10},
            {'params': model.D_domain.parameters(), 'lr': lr_d*5},
#             {'params': model.color_classifier.parameters(), 'lr': lr2*10}
            ], weight_decay=weight_decay, momentum=0.9, nesterov=True)
    
    
#     dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
#     gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
#     model_scheduler = get_scheduler(self.model_opt, hyperparameters)
    
    
    
#     model_opt.zero_grad()


    """ Initialize Writer """
    writer = SummaryWriter()

    best_rank1 = 0
    id_criterion = nn.CrossEntropyLoss()

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
        dis_opt.zero_grad()
        gen_opt.zero_grad()
#        

        """ Train Target Data """
        try:
            _, batch = next(target_iter)
#             print(len(batch['image']))
            if len(batch['image'])< args.batch_size:
                target_iter = enumerate(target_loader)
                _, batch = next(target_iter)
        except:
            target_iter = enumerate(target_loader)
            _, batch = next(target_iter)
        
        images_a = batch['image'][:int(args.batch_size/2)].cuda().view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        images_b = batch['image'][int(args.batch_size/2):].cuda().view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        pos_a = batch['pos_image'][:int(args.batch_size/2)].cuda().view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        pos_b = batch['pos_image'][int(args.batch_size/2):].cuda().view(-1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        
        l_a = batch['label'][:int(args.batch_size/2)].view(-1).cuda()
        l_b = batch['label'][int(args.batch_size/2):].view(-1).cuda()
        
        """ Model Return """
#         target_dict = model(image)
        
        # x_ab, x_ba, c_a, c_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p = model(images_a, images_b, pos_a, pos_b)
        # x_ab, x_ba, s_a, s_b, c_a, c_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, sp_a, sp_b, ssp_a, ssp_b, sc_a, sc_b\
        #      = model(images_a, images_b, pos_a, pos_b)

        x_ab, x_ba, s_a, s_b, c_a, c_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, sp_a, sp_b, ssp_a, ssp_b, sc_a, sc_b\
            , sa_tri, sb_tri, sca_tri, scb_tri, ssa_tri, ssb_tri\
             = model(images_a, images_b, pos_a, pos_b)
        
        
#         print(target_dict['cls_vector'].data.cpu().numpy().shape)

        """ Target Training Loss """
        
        hyperparameters = {
            'recon_s_w': 0,                     # the initial weight for structure code reconstruction
            'recon_f_w': 0,                     # the initial weight for appearance code reconstruction
            'recon_id_w': 0.5,                  # the initial weight for ID reconstruction
            'recon_x_cyc_w': 0,                 # the initial weight for cycle reconstruction
            'recon_x_w': 5,                     # the initial weight for self-reconstruction
            'recon_xp_w': 5,                    # the initial weight for self-identity reconstruction
            'train_bn': True,                   # whether we train the bn for the generated image.
            'use_decoder_again': True,          # whether we train the decoder on the generatd image.
            'use_encoder_again': 0.5,           # the probability we train the structure encoder on the generatd image.
            'vgg_w': 0,                         # We do not use vgg as one kind of inception loss.
            'warm_iter': 30000,                 # when to start warm up the losses (fine-grained/feature reconstruction losses).
            'warm_scale': 0.0005,               # how fast to warm up
            'warm_teacher_iter': 30000,         # when to start warm up the prime loss
            'weight_decay': 0.0005,             # weight decay
            'max_cyc_w': 2,                     # the maximum weight for cycle loss
            'max_iter': 100000,                 # When you end the training
            'max_teacher_w': 2,                 # the maximum weight for prime loss (teacher KL loss)
            'max_w': 1,                       # the maximum weight for feature reconstruction losses
            'gan_w': 1,
            'pid_w': 1.0,
            'id_w': 1.0
        }

##########################################################################################################################################
#Update Generator
##########################################################################################################################################

        # ppa, ppb is the same person
        
        gen_a = gen_b = model.generator
        # id_a = id_b = model.color_extractor
        color_a = color_b = model.color_extractor
        # id_a = id_b = model.structure_extractor

        # no gradient
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)

        rand_num = random.uniform(0,1)
        #################################
        #encode structure
        if hyperparameters['use_encoder_again']>=rand_num:
            # encode again (encoder is tuned, input is fixed)
            s_a_recon = gen_b.enc_content(x_ab_copy)
            s_b_recon = gen_a.enc_content(x_ba_copy)
        else:
            # copy the encoder
            enc_content_copy = copy.deepcopy(gen_a.enc_content)
            enc_content_copy = enc_content_copy.eval()
            # encode again (encoder is fixed, input is tuned)
            s_a_recon = enc_content_copy(x_ab)
            s_b_recon = enc_content_copy(x_ba)
        
        #encode structure
        # if hyperparameters['use_encoder_again']>=rand_num:
        #     # encode again (encoder is tuned, input is fixed)
        #     c_a_recon, _, _ = model.color_extractor(x_ab_copy)
        #     c_b_recon, _, _ = model.color_extractor(x_ba_copy)
        # else:
        #     # copy the encoder
        #     enc_content_copy = copy.deepcopy(model.color_extractor)
        #     enc_content_copy = enc_content_copy.eval()
        #     # encode again (encoder is fixed, input is tuned)
        #     c_a_recon, _, _ = enc_content_copy(x_ab)
        #     c_b_recon, _, _ = enc_content_copy(x_ba)

        #################################


        # encode appearance
        color_a_copy = copy.deepcopy(color_a)
        color_a_copy = color_a_copy.eval()
        if hyperparameters['train_bn']:
            color_a_copy = color_a_copy.apply(train_bn)
        color_b_copy = color_a_copy
        
        # encode again (encoder is fixed, input is tuned)
        c_a_recon, p_a_recon, ff_a_recon = color_a_copy((x_ba))
        c_b_recon, p_b_recon, ff_b_recon = color_b_copy((x_ab))
    
        # p_a_recon = model.structure_classifier(p_a_recon)
        # p_b_recon = model.structure_classifier(p_b_recon)

        # import pdb
        # pdb.set_trace()
        

        # auto-encoder image reconstruction
        x_a = images_a
        x_b = images_b
        loss_gen_recon_x_a = recon_criterion(x_a_recon, x_a)
        loss_gen_recon_x_b = recon_criterion(x_b_recon, x_b)
        loss_gen_recon_xp_a = recon_criterion(x_a_recon_p, x_a)
        loss_gen_recon_xp_b = recon_criterion(x_b_recon_p, x_b)

        # feature reconstruction
        loss_gen_recon_s_a = recon_criterion(s_a_recon, s_a) if hyperparameters['recon_s_w'] > 0 else 0
        loss_gen_recon_s_b = recon_criterion(s_b_recon, s_b) if hyperparameters['recon_s_w'] > 0 else 0
        loss_gen_recon_c_a = recon_criterion(c_a_recon, c_a) if hyperparameters['recon_f_w'] > 0 else 0
        loss_gen_recon_c_b = recon_criterion(c_b_recon, c_b) if hyperparameters['recon_f_w'] > 0 else 0

        x_aba = gen_a.decode(s_a_recon, c_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = gen_b.decode(s_b_recon, c_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # ID loss AND Tune the Generated image
        
        # loss_id = id_criterion(p_a, l_a) + id_criterion(p_b, l_b)
        
        # loss_pid = id_criterion(pp_a, l_a) + id_criterion(pp_b, l_b)
        
        # loss_gen_recon_id = id_criterion(p_a_recon, l_a) + id_criterion(p_b_recon, l_b)

        sp_a_recon, _ = model.structure_classifier(s_a_recon)
        sp_b_recon, _ = model.structure_classifier(s_b_recon)

        loss_id = id_criterion(sp_a, l_a) + id_criterion(sp_b, l_b) + id_criterion(p_a, l_a) + id_criterion(p_b, l_b)
        
        loss_pid = id_criterion(ssp_a, l_a) + id_criterion(ssp_b, l_b) + id_criterion(pp_a, l_a) + id_criterion(pp_b, l_b)
        
        loss_gen_recon_id = id_criterion(sp_a_recon, l_a) + id_criterion(sp_b_recon, l_b) + id_criterion(p_a_recon, l_a) + id_criterion(p_b_recon, l_b)

        # loss_tri = loss_triplet(sp_a, l_a) + loss_triplet(sp_b, l_b)

        # loss_ptri = loss_triplet(ssp_a, l_a) + loss_triplet(ssp_b, l_b)


        # loss_tri = loss_triplet(sp_a, l_a) + loss_triplet(sp_b, l_b)
        loss_tri = loss_triplet(torch.cat([sa_tri,sb_tri,sca_tri,scb_tri]), torch.cat([l_a,l_b,l_a,l_b]))

        # loss_ptri = loss_triplet(ssp_a, l_a) + loss_triplet(ssp_b, l_b)
        # loss_ptri = loss_triplet(ssp_a, l_a) + loss_triplet(ssp_b, l_b)
        loss_ptri = loss_triplet(torch.cat([ssa_tri,ssb_tri]), torch.cat([l_a,l_b]))



        #################### Adversarial learning with GRL #####################
        source_label = 0 # color
        target_label = 1 # gray

        # import pdb
        # pdb.set_trace()

        features_s = grad_reverse(torch.cat([sc_a,sc_b]))
        D_img_out_s = model.D_domain(features_s)
        loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).cuda())

        features_t = grad_reverse(torch.cat([s_a,s_b]))
        # features_t = grad_reverse(features_t['p2'])
        D_img_out_t = model.D_domain(features_t)
        loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).cuda())


        ####################
      
        #print(f_a_recon, f_a)
        loss_gen_cycrecon_x_a = recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        loss_gen_cycrecon_x_b = recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
#         if num_gpu>1:
#             loss_gen_adv_a = dis_a.module.calc_gen_loss(self.dis_a, x_ba)
#             loss_gen_adv_b = dis_b.module.calc_gen_loss(self.dis_b, x_ab)
#         else:
        loss_gen_adv_a = dis_a.calc_gen_loss(dis_a, x_ba)
        loss_gen_adv_b = dis_b.calc_gen_loss(dis_b, x_ab)
            
#         # domain-invariant perceptual loss
#         self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
#         self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        
        if step > hyperparameters['warm_iter']:
            hyperparameters['recon_f_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_f_w'] = min(hyperparameters['recon_f_w'], hyperparameters['max_w'])
            hyperparameters['recon_s_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_s_w'] = min(hyperparameters['recon_s_w'], hyperparameters['max_w'])
            hyperparameters['recon_x_cyc_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_x_cyc_w'] = min(hyperparameters['recon_x_cyc_w'], hyperparameters['max_cyc_w'])
    
        
         # total loss
        loss_gen_total = hyperparameters['gan_w'] * loss_gen_adv_a + \
                              hyperparameters['gan_w'] * loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * loss_gen_recon_x_a + \
                              hyperparameters['recon_xp_w'] * loss_gen_recon_xp_a + \
                              hyperparameters['recon_f_w'] * loss_gen_recon_c_a + \
                              hyperparameters['recon_s_w'] * loss_gen_recon_s_a + \
                              hyperparameters['recon_x_w'] * loss_gen_recon_x_b + \
                              hyperparameters['recon_xp_w'] * loss_gen_recon_xp_b + \
                              hyperparameters['recon_f_w'] * loss_gen_recon_c_b + \
                              hyperparameters['recon_s_w'] * loss_gen_recon_s_b + \
                              hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_b + \
                              hyperparameters['id_w'] * loss_id + \
                              hyperparameters['pid_w'] * loss_pid + \
                              hyperparameters['id_w'] * loss_tri + \
                              hyperparameters['pid_w'] * loss_ptri + \
                              hyperparameters['recon_id_w'] * loss_gen_recon_id + \
                              hyperparameters['gan_w'] * loss_D_img_s + \
                              hyperparameters['gan_w'] * loss_D_img_t
#                               hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
#                               hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
#                               hyperparameters['teacher_w'] * self.loss_teacher
        
        loss_gen_total.backward()
        gen_opt.step()
        model_opt.step()
        
        print("{:6d}/{:6d}| L_total:{:.4f}, L_gan:{:.4f}, Lx:{:.4f}, Lxp:{:.4f}, Lrecycle:{:.4f}, Lf:{:.4f}, Ls:{:.4f}, Recon-id:{:.4f}, id:{:.4f}, pid:{:.4f}, triID:{:.4f}, triPID:{:.4f}, Domain:{:.4f}".format(step+1, args.num_steps,loss_gen_total, \
              hyperparameters['gan_w'] * (loss_gen_adv_a + loss_gen_adv_b), \
              hyperparameters['recon_x_w'] * (loss_gen_recon_x_a + loss_gen_recon_x_b), \
              hyperparameters['recon_xp_w'] * (loss_gen_recon_xp_a + loss_gen_recon_xp_b), \
              hyperparameters['recon_x_cyc_w'] * (loss_gen_cycrecon_x_a + loss_gen_cycrecon_x_b), \
              hyperparameters['recon_f_w'] * (loss_gen_recon_c_a + loss_gen_recon_c_b), \
              hyperparameters['recon_s_w'] * (loss_gen_recon_s_a + loss_gen_recon_s_b), \
              hyperparameters['recon_id_w'] * loss_gen_recon_id/4, \
              hyperparameters['id_w'] * loss_id/4,\
              hyperparameters['pid_w'] * loss_pid/4,\
              hyperparameters['id_w'] * loss_tri/2,\
              hyperparameters['pid_w'] * loss_ptri/2,\
              hyperparameters['gan_w'] * (loss_D_img_t+loss_D_img_s)))
        
        #Write images out to TB: x_a, x_a_recon, x_aba, x_ab1, x_b, x_b_recon, x_bab, x_ba1
        """ Visualize Generated Target Image """
        if (step+1) % args.image_steps == 0:
#             save_model(args, model, D_resolution, D_ACGAN=D_ACGAN) # To save model

            writer.add_image('x1 input', make_grid(tensor2ims(x_a.detach()), nrow=16), step+1)
            writer.add_image('x2 input', make_grid(tensor2ims(x_b.detach()), nrow=16), step+1)
            writer.add_image('pos1 input', make_grid(tensor2ims(pos_a.detach()), nrow=16), step+1)
            writer.add_image('pos2 input', make_grid(tensor2ims(pos_b.detach()), nrow=16), step+1)
            writer.add_image('x12', make_grid(tensor2ims(x_ab.detach()), nrow=16), step+1)
            writer.add_image('x21', make_grid(tensor2ims(x_ba.detach()), nrow=16), step+1)
            writer.add_image('x1_recon', make_grid(tensor2ims(x_a_recon.detach()), nrow=16), step+1)
            writer.add_image('x2_recon', make_grid(tensor2ims(x_b_recon.detach()), nrow=16), step+1)
            writer.add_image('x1_recon_p', make_grid(tensor2ims(x_a_recon_p.detach()), nrow=16), step+1)
            writer.add_image('x2_recon_p', make_grid(tensor2ims(x_b_recon_p.detach()), nrow=16), step+1)
#             writer.add_image('x121', make_grid(tensor2ims(x_aba.detach()), nrow=16), step+1)
#             writer.add_image('x212', make_grid(tensor2ims(x_bab.detach()), nrow=16), step+1)
            
##########################################################################################################################################
#Update Discriminator
##########################################################################################################################################

        dis_opt.zero_grad()
        # D loss
#         if num_gpu>1:
#             self.loss_dis_a, reg_a = self.dis_a.module.calc_dis_loss(self.dis_a, x_ba.detach(), x_a)
#             self.loss_dis_b, reg_b = self.dis_b.module.calc_dis_loss(self.dis_b, x_ab.detach(), x_b)
#         else:
        loss_dis_a, reg_a = dis_a.calc_dis_loss(dis_a, x_ba.detach(), x_a)
        loss_dis_b, reg_b = dis_b.calc_dis_loss(dis_b, x_ab.detach(), x_b)
        loss_dis_total = hyperparameters['gan_w'] * loss_dis_a + hyperparameters['gan_w'] * loss_dis_b
        print("DLoss: {:.4f}".format(loss_dis_total), "Reg: {:.4f}".format(reg_a+reg_b))
        
#         if self.fp16:
#             with amp.scale_loss(self.loss_dis_total, self.dis_opt) as scaled_loss:
#                 scaled_loss.backward()
#         else:
        loss_dis_total.backward()
        dis_opt.step()
        
        """ Write Scalar """
        writer.add_scalar('GAN Loss', hyperparameters['gan_w'] * (loss_gen_adv_a + loss_gen_adv_b), step+1)
        writer.add_scalar('Rec x Loss', hyperparameters['recon_x_w'] * (loss_gen_recon_x_a + loss_gen_recon_x_b), step+1)
        writer.add_scalar('Rec xp Loss', hyperparameters['recon_xp_w'] * (loss_gen_recon_xp_a + loss_gen_recon_xp_b), step+1)
        writer.add_scalar('Rec xcycle Loss', hyperparameters['recon_x_cyc_w'] * (loss_gen_cycrecon_x_a + loss_gen_cycrecon_x_b), step+1)
        writer.add_scalar('Rec f Loss', hyperparameters['recon_f_w'] * (loss_gen_recon_c_a + loss_gen_recon_c_b), step+1)
        writer.add_scalar('Rec s Loss', hyperparameters['recon_s_w'] * (loss_gen_recon_s_a + loss_gen_recon_s_b), step+1)
        writer.add_scalar('Rec id Loss', hyperparameters['recon_id_w'] * loss_gen_recon_id/2, step+1)
        writer.add_scalar('ID Loss', hyperparameters['id_w'] * loss_id/4, step+1)
        writer.add_scalar('PID Loss', hyperparameters['pid_w'] * loss_pid/4, step+1)
        writer.add_scalar('Tri Loss', hyperparameters['id_w'] * loss_tri/2, step+1)
        writer.add_scalar('PTri Loss', hyperparameters['pid_w'] * loss_ptri/2, step+1)
        
        # import pdb
        # pdb.set_trace()
        # if (step+1) % args.eval_steps == 0:
        #     print('Start evaluation...')
 
        #     model.eval()
        #     #rank1 = eval_metric(args, model, test_loader, query_loader)
        #     mAP, cmc, _, _ = eval_metric(args, model, test_loader, query_loader, re_rank=False)
        #     rank1, rank5, rank10, rank20 = cmc[[0,4,9,19]]
            
        #     writer.add_scalar('Rank 1', rank1, (step+1)/args.eval_steps)
        #     writer.add_scalar('Rank 5', rank5, (step+1)/args.eval_steps)
        #     writer.add_scalar('Rank 10', rank10, (step+1)/args.eval_steps)
        #     writer.add_scalar('Rank 20', rank20, (step+1)/args.eval_steps)
        #     writer.add_scalar('mAP', mAP, (step+1)/args.eval_steps)

        #     if rank1 >= best_rank1:
        #         best_rank1 = rank1
        #         print('Saving model...')
        #         save_model(args, model, dis_model) # To save model
        #         writer.add_scalar('Best Rank 1', best_rank1, (step+1)/args.eval_steps)

        #     print('Rank:', rank1, rank5, rank10, rank20)
        #     print('mAP:', mAP)
        #     print('Best rank1:', best_rank1)


if __name__ == '__main__':
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    main()
