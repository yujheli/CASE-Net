import os
import config

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable, grad
from model.network import AdaptVAEReID
from model.discriminator import Discriminator
from model.discriminator import ACGAN

from data.duke import Duke
from data.market import Market
from data.msmt import MSMT
from data.cuhk import CUHK
from data.viper import VIPER
from data.caviar import CAVIAR

from util.dataloader import DataLoader
from util.normalize import NormalizeImage


inv_normalize = transforms.Normalize(
    mean=[-1.0, -1.0, -1.0],
    std=[2.0, 2.0, 2.0]
)


def calc_gradient_penalty(netD, real_data, fake_data, use_gpu = True, dec_output=2):
    alpha = torch.rand(real_data.shape[0], 1)
    if len(real_data.shape) == 4:
        alpha = alpha.unsqueeze(2).unsqueeze(3)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_gpu else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_gpu:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    if dec_output==2:
        disc_interpolates,_ = netD(interpolates)
    elif dec_output == 3:
        disc_interpolates,_,_ = netD(interpolates)
    else:
        disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                    grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_gpu else torch.ones(
                                  disc_interpolates.size()),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def make_one_hot(labels, args):
    
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
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(-1)
    one_hot = torch.cuda.FloatTensor(labels.size(0), classifier_output_dim).zero_()
    target = one_hot.scatter_(1, labels.cuda(), 1)
    
    target = Variable(target)
        
    return target



def save_model(args, model, D_1=None, D_2=None, D_ACGAN=None):

    extractor_path = os.path.join(args.model_dir, 'Extractor_{}.pth.tar'.format(args.source_dataset))
    decoder_path = os.path.join(args.model_dir, 'Decoder_{}.pth.tar'.format(args.source_dataset))
    classifier_path = os.path.join(args.model_dir, 'Classifier_{}.pth.tar'.format(args.source_dataset))
    var_path = os.path.join(args.model_dir, 'Var_{}.pth.tar'.format(args.source_dataset))
    
    if D_1:
        D1_path = os.path.join(args.model_dir, 'D1_{}.pth.tar'.format(args.source_dataset))
    if D_2:
        D2_path = os.path.join(args.model_dir, 'D2_{}.pth.tar'.format(args.source_dataset))
    if D_ACGAN:
        ACGAN_path = os.path.join(args.model_dir, 'ACGAN_{}.pth.tar'.format(args.source_dataset))

    torch.save(model.extractor.state_dict(), extractor_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    torch.save(model.classifier.state_dict(), classifier_path)
    torch.save(model.encode_mean_logvar.state_dict(), var_path)
    
    if D_1:
        torch.save(D_1.state_dict(), D1_path)
    if D_2:
        torch.save(D_2.state_dict(), D2_path)
    if D_ACGAN:
        torch.save(D_ACGAN.state_dict(), ACGAN_path)

    return


def init_model(args, use_cuda=True):

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
                         classifier_output_dim=classifier_output_dim,
                         code_dim=classifier_output_dim)

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
            
    if args.var_path:
        print("Loading pre-trained var...")
        checkpoint = torch.load(args.var_path, map_location=lambda storage, loc: storage)
        for name, param in model.encode_mean_logvar.state_dict().items():
            model.encode_mean_logvar.state_dict()[name].copy_(checkpoint[name])

    return model


def init_resolution_D(args, use_cuda=True):

    model = Discriminator(input_channel=config.D1_INPUT_CHANNEL,
                          fc_input_dim=config.D1_FC_INPUT_DIM,
                          use_cuda=use_cuda)

    if args.discriminator_path:
        print("Loading pre-trained resolution discriminator...")
        checkpoint = torch.load(args.discriminator_path, map_location=lambda storage, loc: storage)
        for name, param in model.state_dict().items():
            model.state_dict()[name].copy_(checkpoint[name])

    return model


def init_ACGAN(args, use_cuda=True):

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

    model = ACGAN(fc_input_dim=config.D1_FC_INPUT_DIM,
                  class_num=classifier_output_dim,
                  use_cuda=use_cuda)

    if args.acgan_path:
        print("Loading pre-trained ACGAN discriminator...")
        checkpoint = torch.load(args.acgan_path, map_location=lambda storage, loc: storage)
        for name, param in model.state_dict().items():
            model.state_dict()[name].copy_(checkpoint[name])

    return model


def init_source_data(args):

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

    source_data = SourceData(mode='source',
                             transform=NormalizeImage(['image', 'rec_image']),
                             random_crop=args.random_crop)

    source_loader = DataLoader(source_data,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=True)

    return source_data, source_loader


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


    target_data = TargetData(mode='train',
                             transform=NormalizeImage(['image', 'rec_image']),
                             random_crop=args.random_crop)

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


    test_data = TestData(mode='test',
                         transform=NormalizeImage(['image']))

    test_loader = DataLoader(test_data,
                             batch_size=int(args.batch_size/2),
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


    query_data = QueryData(mode='query',
                           transform=NormalizeImage(['image']))

    query_loader = DataLoader(query_data,
                              batch_size=int(args.batch_size/2),
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


def init_D_optim(args, model):
    
    model_opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.learning_rate / 10.0,
                           betas=(0.9, 0.99))

    model_opt.zero_grad()

    return model_opt

