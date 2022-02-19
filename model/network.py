from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import config
import torch.nn.init as init
import cv2
from .discriminator import *

############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################


################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score
#         return cls_score
    

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def recover(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(config.MEAN)
    std = np.array(config.STDDEV)
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    inp = inp.astype(np.uint8)
    return inp

# def to_grays(x):
#     x = x.data.cpu()
#     out = torch.FloatTensor(x.size(0), x.size(1), x.size(2), x.size(3))
#     mean = np.array(config.MEAN)
#     std = np.array(config.STDDEV)
#     for i in range(x.size(0)):
#         xx = recover(x[i,:,:,:])   # 3 channel, 256x128x3
#         xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY) # 256x128x1
# #         xx = cv2.Canny(xx, 10, 200) #256x128
#         xx = xx/255.0 - 0.5 # {-0.5,0.5}
# #         xx = xx/255.0
# #         xx += np.random.randn(xx.shape[0],xx.shape[1])*0.1  #add random noise
#         xx = torch.from_numpy(xx.astype(np.float32))
#         out[i,0,:,:] = xx
#         out[i,1,:,:] = xx
#         out[i,2,:,:] = xx
# #     out = out.unsqueeze(1) 
#     return out.cuda()

def to_grays(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(1), x.size(2), x.size(3))
    mean = np.array(config.MEAN)
    std = np.array(config.STDDEV)
    for i in range(x.size(0)):
        xx = recover(x[i,:,:,:])   # 3 channel, 256x128x3
        gray = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY) # 256x128x1
        img2 = np.zeros_like(xx)
        img2[:,:,0] = gray
        img2[:,:,1] = gray
        img2[:,:,2] = gray
#         xx = cv2.Canny(xx, 10, 200) #256x128
#         xx = xx/255.0 - 0.5 # {-0.5,0.5}
#         xx = xx/255.0
#         xx = (xx-mean)/std
#         xx += np.random.randn(xx.shape[0],xx.shape[1])*0.1  #add random noise
        img2 = (img2-mean)/std/255.
        xx = torch.from_numpy(img2.transpose((2,0,1)).astype(np.float32))
        out[i,:,:,:] = xx
#         out[i,1,:,:] = xx
#         out[i,2,:,:] = xx
#     out = out.unsqueeze(1) 
    return out.cuda()

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        #num_bottleneck = input_dim # We remove the input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck, affine=True)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params, fp16):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.LAMBDA = params['LAMBDA']
        self.non_local = params['non_local']
        self.n_res = params['n_res']
        self.input_dim = input_dim
        self.fp16 = fp16
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        if not self.gan_type == 'wgan':
            self.cnns = nn.ModuleList()
            for _ in range(self.num_scales):
                Dis = self._make_net()
                Dis.apply(weights_init('gaussian'))
                self.cnns.append(Dis)
        else:
             self.cnn = self.one_cnn()

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 1, 1, 0, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            dim2 = min(dim*2, 512)
            cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            cnn_x += [Conv2dBlock(dim, dim2, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = dim2
        if self.non_local>1:
            cnn_x += [NonlocalBlock(dim)]
        for i in range(self.n_res):
            cnn_x += [ResBlock(dim, norm=self.norm, activation=self.activ, pad_type=self.pad_type, res_type='basic')] 
        if self.non_local>0:
            cnn_x += [NonlocalBlock(dim)]
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def one_cnn(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(5):
            dim2 = min(dim*2, 512)
            cnn_x += [Conv2dBlock(dim, dim2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = dim2
        cnn_x += [nn.Conv2d(dim, 1, (4,2), 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        if not self.gan_type == 'wgan':
            outputs = []
            for model in self.cnns:
                outputs.append(model(x))
                x = self.downsample(x)
        else:
            outputs = self.cnn(x)
            outputs = torch.squeeze(outputs)
        return outputs

    def calc_dis_loss(self, model, input_fake, input_real):
        # calculate the loss to train D
        input_real.requires_grad_()
        outs0 = model.forward(input_fake)
        outs1 = model.forward(input_real)
        loss = 0
        reg = 0
        Drift = 0.001
        LAMBDA = self.LAMBDA

        if self.gan_type == 'wgan':
            loss += torch.mean(outs0) - torch.mean(outs1)
            # progressive gan
            loss += Drift*( torch.sum(outs0**2) + torch.sum(outs1**2))
            #alpha = torch.FloatTensor(input_fake.shape).uniform_(0., 1.)
            #alpha = alpha.cuda()
            #differences = input_fake - input_real
            #interpolates =  Variable(input_real + (alpha*differences), requires_grad=True)
            #dis_interpolates = self.forward(interpolates) 
            #gradient_penalty = self.compute_grad2(dis_interpolates, interpolates).mean()
            #reg += LAMBDA*gradient_penalty 
            reg += LAMBDA* self.compute_grad2(outs1, input_real).mean() # I suggest Lambda=0.1 for wgan
            loss = loss + reg
            return loss, reg

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
                # regularization
                reg += LAMBDA* self.compute_grad2(out1, input_real).mean()
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
                reg += LAMBDA* self.compute_grad2(F.sigmoid(out1), input_real).mean()
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss = loss+reg
        return loss, reg
    
    def calc_gen_loss(self, model, input_fake):
        # calculate the loss to train G
        outs0 = model.forward(input_fake)
        loss = 0
        Drift = 0.001
        if self.gan_type == 'wgan':
            loss += -torch.mean(outs0)
            # progressive gan
            loss += Drift*torch.sum(outs0**2)
            return loss

        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) * 2  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg


def to_gray(half=False): #simple
    def forward(x):
        x = torch.mean(x, dim=1, keepdim=True)
        if half:
            x = x.half()
        return x
    return forward

def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = recover(x[i,:,:,:])   # 3 channel, 256x128x3
        xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY) # 256x128x1
        xx = cv2.Canny(xx, 10, 200) #256x128
        xx = xx/255.0 - 0.5 # {-0.5,0.5}
        xx += np.random.randn(xx.shape[0],xx.shape[1])*0.1  #add random noise
        xx = torch.from_numpy(xx.astype(np.float32))
        out[i,:,:] = xx
    out = out.unsqueeze(1) 
    return out.cuda()

class Strong_ReID(nn.Module):

    def __init__(self, 
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION,
                 classifier_input_dim=2048,
                 classifier_output_dim=config.DUKE_CLASS_NUM,
                 use_cuda=True,
                 local_conv_out_channels=128):
        super(Strong_ReID, self).__init__()

        self.class_num = classifier_output_dim

        # backbone and optimize its architecture
        resnet = models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = BNClassifier(2048, self.class_num)

    def forward(self, x):

        features = self.gap(self.resnet_conv(x)).squeeze()
        bned_features, cls_score = self.classifier(features)

#         if self.training:
#             return features, cls_score
#         else:
#             return bned_features
        
        return_dict = {'cls_vector': cls_score, 
                       'global_feature': features,
                       'local_feature': bned_features}
        return return_dict

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 4 # We cut the pool5 to 4 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.softmax = nn.Softmax(dim=1)
        # define 4 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        f = x
        f = f.view(f.size(0),f.size(1)*self.part)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get part feature batchsize*2048*4
        for i in range(self.part):
            part[i] = x[:,:,i].contiguous()
            part[i] = part[i].view(x.size(0), x.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        y=[]
        for i in range(self.part):
            y.append(predict[i])

        return f, y

class PCB_ReID(nn.Module):
    def __init__(self,
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION,
                 classifier_input_dim=2048,
                 classifier_output_dim=config.DUKE_CLASS_NUM,
                 use_cuda=True,
                 local_conv_out_channels=128):
        super(PCB_ReID, self).__init__()
        
        self.part = 1
        self.extractor = PCB_extractor(backbone=backbone, stride=1, part_num=self.part)

        self.skip_connection = skip_connection

#         self.classifier = Classifier(input_dim=classifier_input_dim,
#                                      output_dim=classifier_output_dim)

        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, BNClassifier(classifier_input_dim, classifier_output_dim))

        if use_cuda:
            self.extractor = self.extractor.cuda()
#             self.classifier = self.classifier.cuda()
            for i in range(self.part):
                name = 'classifier'+str(i)
                c = getattr(self,name)
                c = c.cuda()
    
 
    def forward(self, data):
        
        x, feat = self.extractor(data)
        part = {}
        predict = {}
        local_predict = {}
        local_f = []
        # get part feature batchsize*2048*4
        for i in range(self.part):
            part[i] = x[:,:,i].contiguous()
            part[i] = part[i].view(x.size(0), x.size(1))
#             local_f.append(part[i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            local_predict[i], predict[i]= c(part[i])
            local_f.append(local_predict[i])

        y=[]
        for i in range(self.part):
            y.append(predict[i])

#         local_feature = x.view(feat.size(0),feat.size(1),self.part)

        return_dict = {'cls_vectors': y, 
                       'global_feature': feat,
                       'local_feature': local_f}

        return return_dict
    
    
class Sense_ReID(nn.Module):
    def __init__(self,
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION,
                 classifier_input_dim=2048,
                 classifier_output_dim=config.DUKE_CLASS_NUM,
                 use_cuda=True,
                 local_conv_out_channels=128):
        super(Sense_ReID, self).__init__()

#         self.structure_extractor = Extractor_v2(backbone=backbone)
        # self.structure_extractor = ft_netAB(backbone=backbone)

        self.color_extractor = Extractor_v2(backbone=backbone,class_num=classifier_output_dim)
        # self.color_extractor = ft_net(class_num=classifier_output_dim)
        
        # self.color_extractor = ft_netAB(backbone=backbone)


        self.skip_connection = skip_connection

        # self.structure_classifier = Classifier(input_dim=classifier_input_dim,
        #                              output_dim=classifier_output_dim)
        self.structure_classifier = Classifier(input_dim=128,
                                     output_dim=classifier_output_dim)

        self.D_domain = FCDiscriminator_img(num_classes=128) # Need to know the channel
        # self.structure_classifier = Classifier(input_dim=128,
        #                              output_dim=classifier_output_dim)
        
#         self.color_classifier = Classifier(input_dim=classifier_input_dim,
#                                      output_dim=classifier_output_dim)
        
        input_dim_a = 3
        gen_params ={
          'activ': 'lrelu',                   # activation function style [relu/lrelu/prelu/selu/tanh]
          'dec': 'basic',                     # [basic/parallel/series]
          'dim': 16,                          # number of filters in the bottommost layer
          'dropout': 0,                       # use dropout in the generator
          'id_dim': 2048,                     # length of appearance code
          'mlp_dim': 512,                     # number of filters in MLP
          'mlp_norm': 'none',                 # norm in mlp [none/bn/in/ln]
          'n_downsample': 2,                  # number of downsampling layers in content encoder
          'n_res': 4,                         # number of residual blocks in content encoder/decoder
          'non_local': 0,                     # number of non_local layer
          'pad_type': 'reflect',              # padding type [zero/reflect]
          'tanh': False,                    # use tanh or not at the last layer
          'init': 'kaiming',                  # initialization [gaussian/kaiming/xavier/orthogonal]
        }
        
        self.generator = AdaINGen(input_dim_a, gen_params, fp16 = False)  # auto-encoder for domain a
        self.gen_a = self.generator
        self.gen_b = self.generator
        
        
        if use_cuda:
            # self.structure_extractor = self.structure_extractor.cuda()
            self.color_extractor = self.color_extractor.cuda()
            self.structure_classifier = self.structure_classifier.cuda()
            self.generator = self.generator.cuda()
            self.D_domain = self.D_domain.cuda()
        self.color_a = self.color_b = self.color_extractor
        
    
    def forward(self, images_a, images_b, pos_a, pos_b):

        sc_a = self.gen_a.encode((images_a)) # structure color fearture
        sc_b = self.gen_b.encode((images_b)) # structure color feature


        s_a = self.gen_a.encode(to_grays(images_a)) # structure gray fearture
        s_b = self.gen_b.encode(to_grays(images_b)) # structure gray feature

        c_a, p_a , _  = self.color_a(images_a) #Extract the color
        c_b, p_b , _ = self.color_b(images_b)


        # f_a, p_a = self.id_a(scale2(x_a))
        # f_b, p_b = self.id_b(scale2(x_b))

        x_ba = self.gen_a.decode(s_b, c_a)
        x_ab = self.gen_b.decode(s_a, c_b)

        x_a_recon = self.gen_a.decode(s_a, c_a)
        x_b_recon = self.gen_b.decode(s_b, c_b)


        # fp_a, pp_a = self.id_a(scale2(xp_a))
        # fp_b, pp_b = self.id_b(scale2(xp_b))

        cp_a, pp_a , _  = self.color_a(pos_a) #Extract the color
        cp_b, pp_b , _ = self.color_b(pos_b)



        # # decode the same person
        x_a_recon_p = self.gen_a.decode(s_a, cp_a)
        x_b_recon_p = self.gen_b.decode(s_b, cp_b)
        
        ss_a = self.gen_a.encode(to_grays(pos_a)) # structure fearture
        ss_b = self.gen_b.encode(to_grays(pos_b)) # structure feature

        sp_a, sa_tri = self.structure_classifier(s_a)
        sp_b, sb_tri = self.structure_classifier(s_b)

        _, sca_tri = self.structure_classifier(sc_a)
        _, scb_tri = self.structure_classifier(sc_b)
        
        ssp_a, ssa_tri = self.structure_classifier(ss_a)
        ssp_b, ssb_tri = self.structure_classifier(ss_b)


        # return x_ab, x_ba, c_a, c_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p
        # return x_ab, x_ba, c_a, c_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p
        return x_ab, x_ba, s_a, s_b, c_a, c_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, sp_a, sp_b, ssp_a, ssp_b, sc_a, sc_b, sa_tri, sb_tri, sca_tri, scb_tri, ssa_tri, ssb_tri
        # sc_a, sc_b
        return return_dict

           

##################################################################################
# New modules added start line
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params, fp16):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        mlp_norm = params['mlp_norm']
        id_dim = params['id_dim']
        which_dec = params['dec']
        dropout = params['dropout']
        tanh = params['tanh']
        non_local = params['non_local']

        # content encoder
        # self.enc_content = ContentEncoder_ImageNet()

        # n_downsample = 4
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type, dropout=dropout, tanh=tanh, res_type='basic')

        self.output_dim = self.enc_content.output_dim
        print(self.output_dim)
        # self.output_dim = 128
        
        if which_dec =='basic':        
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='basic', non_local = non_local, fp16 = fp16)
        elif which_dec =='slim':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='slim', non_local = non_local, fp16 = fp16)
        elif which_dec =='series':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='series', non_local = non_local, fp16 = fp16)
        elif which_dec =='parallel':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='parallel', non_local = non_local, fp16 = fp16)
        else:
            ('unkonw decoder type')

        # MLP to generate AdaIN parameters
        self.mlp_w1 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w2 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w3 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w4 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        
        self.mlp_b1 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b2 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b3 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b4 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)

        self.apply(weights_init(params['init']))

    def encode(self, images): # structure
        # encode an image to its content and style codes
        content = self.enc_content(images)
        return content

    def decode(self, content, ID):
        # decode style codes to an image
        # The feature from PCB (?)

        ID1 = ID[:,:2048]
        ID2 = ID[:,2048:4096]
        ID3 = ID[:,4096:6144]
        ID4 = ID[:,6144:]
        adain_params_w = torch.cat( (self.mlp_w1(ID1), self.mlp_w2(ID2), self.mlp_w3(ID3), self.mlp_w4(ID4)), 1)
        adain_params_b = torch.cat( (self.mlp_b1(ID1), self.mlp_b2(ID2), self.mlp_b3(ID3), self.mlp_b4(ID4)), 1)
        self.assign_adain_params(adain_params_w, adain_params_b, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params_w, adain_params_b, model):
        # assign the adain_params to the AdaIN layers in model
        dim = self.output_dim
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params_b[:,:dim].contiguous()
                std = adain_params_w[:,:dim].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1)>dim :  #Pop the parameters
                    adain_params_b = adain_params_b[:,dim:]
                    adain_params_w = adain_params_w[:,dim:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += m.num_features
        return num_adain_params

    

    
class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, dropout=0, res_norm='adain', activ='relu', pad_type='zero', res_type='basic', non_local=False, fp16 = False):
        super(Decoder, self).__init__()
        self.input_dim = dim
        self.model = []
        self.model += [nn.Dropout(p = dropout)]
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type, res_type=res_type)]
        # non-local
        if non_local>0:
            self.model += [NonlocalBlock(dim)]
            print('use non-local!')
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type, fp16 = fp16)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        output = self.model(x)
        return output
##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        # Here I change the stride to 2. 
        self.model += [Conv2dBlock(input_dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, dropout, tanh=False, res_type='basic'):
        super(ContentEncoder, self).__init__()
        self.model = []
        # Here I change the stride to 2.
        self.model += [Conv2dBlock(input_dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, 2*dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim *=2 # 32dim
        # downsampling blocks
        for i in range(n_downsample-1): 
            self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type, res_type=res_type)]
        # 64 -> 128
        self.model += [ASPP(dim, norm=norm, activation=activ, pad_type=pad_type)]
        dim *= 2
        if tanh:
            self.model +=[nn.Tanh()]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder_ImageNet(nn.Module):
    def __init__(self):
        super(ContentEncoder_ImageNet, self).__init__()
        # self.model = models.resnet50(pretrained=True)
        self.model = models.resnet34(pretrained=True)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1) 
        # (256,128) ----> (16,8)
        self.output_dim = 2048
        self.output_dim = 512
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        
        return x

    
##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlocks, self).__init__()
        self.model = []
        self.res_type = res_type
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, res_type=res_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='in', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    

    

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlock, self).__init__()

        model = []
        if res_type=='basic' or res_type=='nonlocal':
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type=='slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim ,dim_half, 1, 1, 0, norm='in', activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', pad_type=pad_type)]
        elif res_type=='series':
            model += [Series2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Series2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type=='parallel':
            model += [Parallel2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Parallel2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)
        if res_type=='nonlocal':
            self.nonloc = NonlocalBlock(dim)

    def forward(self, x):
        if self.res_type == 'nonlocal':
            x = self.nonloc(x)
        residual = x
        out = self.model(x)
        out += residual
        return out

class NonlocalBlock(nn.Module):
    def __init__(self, in_dim, norm='in'):
        super(NonlocalBlock, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class ASPP(nn.Module):
    # ASPP (a)
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ASPP, self).__init__()
        dim_part = dim//2
        self.conv1 = Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)

        self.conv6 = []
        self.conv6 += [Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv6 += [Conv2dBlock(dim_part,dim_part, 3, 1, 3, norm=norm, activation='none', pad_type=pad_type, dilation=3)]
        self.conv6 = nn.Sequential(*self.conv6)

        self.conv12 = []
        self.conv12 += [Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv12 += [Conv2dBlock(dim_part,dim_part, 3, 1, 6, norm=norm, activation='none', pad_type=pad_type, dilation=6)]
        self.conv12 = nn.Sequential(*self.conv12)

        self.conv18 = []
        self.conv18 += [Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv18 += [Conv2dBlock(dim_part,dim_part, 3, 1, 9, norm=norm, activation='none', pad_type=pad_type, dilation=9)]
        self.conv18 = nn.Sequential(*self.conv18)

        self.fuse = Conv2dBlock(4*dim_part,2*dim, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type) 

    def forward(self, x):
        conv1 = self.conv1(x)
        conv6 = self.conv6(x)
        conv12 = self.conv12(x)
        conv18 = self.conv18(x)
        out = torch.cat((conv1,conv6,conv12, conv18), dim=1)
        out = self.fuse(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, fp16 = False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, fp16 = fp16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



class Series2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Series2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x) + x
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class Parallel2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Parallel2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x)) + self.norm(x)
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            #reshape input
            out = out.unsqueeze(1)
            out = self.norm(out)
            out = out.view(out.size(0),out.size(2))
        if self.activation:
            out = self.activation(out)
        return out    


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x    
####################################################################################################################################################################
#New modules ended line
####################################################################################################################################################################

# Define the AB Model
class ft_netAB(nn.Module):
    def __init__(self, backbone='resnet-50', norm=False, stride=2, droprate=0.5, pool='avg'):
        super(ft_netAB, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.part = 4
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft

        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1,1)
            self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x)
        f = f.view(f.size(0),f.size(1)*self.part)
        f = f.detach() # no gradient 
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        
        return f, x

class PCB_extractor(nn.Module):
    def __init__(self, backbone='resnet-50', norm=False, stride=1, droprate=0.5, pool='avg', part_num=4):
        super(PCB_extractor, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.part = part_num
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft

        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1,1)
            self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        
        return f, x    

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, norm=False, pool='avg', stride=2):
        super(ft_net, self).__init__()
        if norm:
            self.norm = True
        else:
            self.norm = False
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        self.part = 4
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1)) 
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1)) 
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # remove the final downsample
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.model = model_ft   
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)  # -> 512 32*16
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x) # 8 * 2048 4*1
        x = self.model.avgpool(x)  # 8 * 2048 1*1
        
        x = x.view(x.size(0),x.size(1))
        f = f.view(f.size(0),f.size(1)*self.part)
        if self.norm:
            fnorm = torch.norm(f, p=2, dim=1, keepdim=True) + 1e-8
            f = f.div(fnorm.expand_as(f))
        x = self.classifier(x)
        return f, x

class Extractor_v2(nn.Module):
    def __init__(self,
                 backbone='resnet-101',class_num=155,
                 skip_connection=config.SKIP_CONNECTION, norm=False, stride=2, droprate=0.5, pool='avg'):
        super(Extractor_v2, self).__init__()
        self.skip_connection = skip_connection
        if backbone == 'resnet-50':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            #self.skip_idx = ['2', '4', '5', '6', '7', '8']
            self.skip_idx = ['2', '4', '5', '6', '7']
        
        elif backbone == 'resnet-101':
            self.model = models.resnet101(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            #self.skip_idx = ['2', '4', '5', '6', '7', '8']
            self.skip_idx = ['2', '4', '5', '6', '7']
        
        self.part = 4
        if pool=='max':
            self.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            self.avgpool = nn.AdaptiveMaxPool2d((1,1))
            
        else:
            self.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fitpool = nn.AdaptiveAvgPool3d((128,64,32))
        self.classifier = Classifier2(2048, class_num)

    def forward(self, data):
        x = self.model(data)
        d = self.fitpool(x.repeat(1,1,8,8))
        f = self.partpool(x)
        f = f.view(f.size(0),f.size(1)*self.part)
        f = f.detach() # no gradient 
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        # return f, x, d
        return f, x, d
    
# class Extractor(nn.Module):
#     def __init__(self,
#                  backbone='resnet-101',
#                  skip_connection=config.SKIP_CONNECTION):
#         super(Extractor, self).__init__()
#         self.skip_connection = skip_connection
#         if backbone == 'resnet-50':
#             self.model = models.resnet50(pretrained=True)
#             self.model = nn.Sequential(*list(self.model.children())[:-2])
#             #self.skip_idx = ['2', '4', '5', '6', '7', '8']
#             self.skip_idx = ['2', '4', '5', '6', '7']
        
#         elif backbone == 'resnet-101':
#             self.model = models.resnet101(pretrained=True)
#             self.model = nn.Sequential(*list(self.model.children())[:-2])
#             #self.skip_idx = ['2', '4', '5', '6', '7', '8']
#             self.skip_idx = ['2', '4', '5', '6', '7']

#     def forward(self, data):
#         skip_features = []
#         for idx, module in self.model._modules.items():
#             data = module(data)
#             if idx in self.skip_idx:
#                 skip_features.append(data)
        
#         return skip_features

class Classifier(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 output_dim=config.DUKE_CLASS_NUM):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.batch_norm =  nn.BatchNorm1d(input_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, data):
        x = data
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        data = x
        data = data.view(data.size()[0],-1)
        data = self.batch_norm(data)
        out = self.linear(data)
        return out, x
    
class Classifier2(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 output_dim=config.DUKE_CLASS_NUM):
        super(Classifier2, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.batch_norm =  nn.BatchNorm1d(input_dim)
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, data):
        # x = data
        # x = self.avgpool(x)
        # x = x.view(x.size(0), x.size(1))
        # data = x
        data = data.view(data.size()[0],-1)
        data = self.batch_norm(data)
        out = self.linear(data)
        return out
    
class AdaptVAEReID(nn.Module):
    pass
#     def __init__(self,
#                  backbone='resnet-101',
#                  skip_connection=config.SKIP_CONNECTION,
#                  classifier_input_dim=2048,
#                  classifier_output_dim=config.DUKE_CLASS_NUM,
#                  use_cuda=True,
#                  local_conv_out_channels=128,mu_dim=2048, code_dim=config.DUKE_CLASS_NUM):
#         super(AdaptVAEReID, self).__init__()

#         self.extractor = Extractor(backbone=backbone)
        
#         self.code_dim = code_dim
#         self.mu_dim = mu_dim
#         self.skip_connection = skip_connection
        
# #         self.enc_mu = nn.Conv2d(classifier_input_dim, self.mu_dim, kernel_size=4, stride=2,padding=1)
# #         self.enc_logvar = nn.Conv2d(classifier_input_dim, self.mu_dim, kernel_size=4, stride=2,padding=1)
#         self.encode_mean_logvar = Encode_Mean_Logvar(input_channel = classifier_input_dim, output_channel = self.mu_dim)

#         self.decoder = Res_Decoder(backbone=backbone, input_dim=self.mu_dim, code_dim=self.code_dim, skip_connection=self.skip_connection)

#         self.classifier = Classifier(input_dim=classifier_input_dim,
#                                      output_dim=classifier_output_dim)

        

#         self.avgpool = nn.AvgPool2d((8,4))
        
#         #local feature
#         self.local_conv = nn.Conv2d(classifier_input_dim, local_conv_out_channels, 1)
#         self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
#         self.local_relu = nn.ReLU()

#         if use_cuda:
#             self.extractor = self.extractor.cuda()
#             self.decoder = self.decoder.cuda()
#             self.classifier = self.classifier.cuda()
#             self.local_conv = self.local_conv.cuda()
#             self.local_bn = self.local_bn.cuda()
#             self.local_relu = self.local_relu.cuda()
# #             self.enc_mu = self.enc_mu.cuda()
# #             self.enc_logvar = self.enc_logvar.cuda()
#             self.encode_mean_logvar = self.encode_mean_logvar.cuda()
            
            
    
#     def decode(self, z, insert_attrs = None, features=None):
        
#         if len(z.size()) != 4:
#             z = z.view(z.size()[0],self.mu_dim,4,2)

#         if insert_attrs is not None:
#             if len(z.size()) == 2:
#                 z = torch.cat([z,insert_attrs],dim=1)
#             else:
#                 H,W = z.size()[2], z.size()[3]
#                 z = torch.cat([z,insert_attrs.unsqueeze(-1).unsqueeze(-1).repeat(1,1,H,W)],dim=1)
# #                 print(z.size())
#         reconstruct, skip_features_dif, skip_features_ori = self.decoder(data=z, features=features)
       
#         return reconstruct, skip_features_dif, skip_features_ori
    
#     def encode(self, x):
# #         for l in range(len(self.enc_layers)-1):
# #             if (self.enc_layers[l] == 'fc')  and (len(x.size())>2):
# #                 batch_size = x.size()[0]
# #                 x = x.view(batch_size,-1)
# #             x = getattr(self, 'enc_'+str(l))(x)

# #         if (self.enc_layers[-1] == 'fc')  and (len(x.size())>2):
# #             batch_size = x.size()[0]
# #             x = x.view(batch_size,-1)
#         features = self.extractor(data=x)
#         extracted_feature = features[-1]
# #         mu = self.enc_mu(extracted_feature)
# #         logvar = self.enc_logvar(extracted_feature)
#         mu, logvar =  self.encode_mean_logvar(extracted_feature)
#         if len(mu.size()) > 2:
#             mu = mu.view(mu.size()[0],-1)
#             logvar = logvar.view(mu.size()[0],-1)

#         return mu, logvar
    

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = Variable(std.data.new(std.size()).normal_())
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
    
 
#     def forward(self, data, insert_attrs=None, return_enc=False):

#         features = self.extractor(data=data)

#         extracted_feature = features[-1]
        
#         mu, logvar =  self.encode_mean_logvar(extracted_feature)
# #         mu = self.enc_mu(extracted_feature)
# #         logvar = self.enc_logvar(extracted_feature)

#         latent_feature = self.avgpool(extracted_feature)

#         cls_vector = self.classifier(data=latent_feature)
        
        
        
#         if len(mu.size()) > 2:
#             mu = mu.view(mu.size()[0],-1)
#             logvar = logvar.view(mu.size()[0],-1)
#         z = self.reparameterize(mu, logvar)
# #         z = Variable(mu.data.new(mu.size()).normal_())

# #         reconstruct = self.decoder(features=features)

#         if self.skip_connection:
#             skip_features = features
#         else:
#             skip_features = None
    
#         if insert_attrs is not None:
#             reconstruct, skip_features_dif, skip_features_ori = self.decode(z, insert_attrs, features=skip_features)    
#         else:
#             reconstruct, skip_features_dif, skip_features_ori = self.decode(z, cls_vector, features=skip_features)
            

        
#         # shape [N, C]
#         global_feat = latent_feature.view(latent_feature.size(0), -1)
#         # shape [N, C, H, 1]
#         local_feat = torch.mean(extracted_feature, -1, keepdim=True)
#         local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
#         # shape [N, H, c]
#         local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

#         #return latent_feature, features, cls_vector, reconstruct, global_feat, local_feat, mu, logvar
#         #return {'latent_vector': latent_feature, 

#         resolution_feature = features[-1]

#         return_dict = {'latent_vector': latent_feature, 
#                        'resolution_feature': resolution_feature, 
#                        'cls_vector': cls_vector, 
#                        'rec_image': reconstruct, 
#                        'global_feature': global_feat, 
#                        'local_feature': local_feat, 
#                        'mu': mu, 
#                        'logvar': logvar,
#                        'image': data,
#                        'features': skip_features_ori,
#                        'skip_e': skip_features_dif }

# #         if insert_attrs is not None:
# #             return_dict['skip_e'] = skip_features_e
# #         else:
# #             return_dict['skip_e'] = cls_vector

#         return return_dict
    
    
class Baseline_ReID(nn.Module):
    def __init__(self,
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION,
                 classifier_input_dim=2048,
                 classifier_output_dim=config.DUKE_CLASS_NUM,
                 use_cuda=True,
                 local_conv_out_channels=128):
        super(Baseline_ReID, self).__init__()

        self.extractor = Extractor(backbone=backbone)
        
#         self.code_dim = code_dim
#         self.mu_dim = mu_dim
        self.skip_connection = skip_connection

        self.classifier = Classifier(input_dim=classifier_input_dim,
                                     output_dim=classifier_output_dim)

        

#         self.avgpool = nn.AvgPool2d((8,4))
        self.avgpool = nn.AdaptiveMaxPool2d((1,1))
        
        #local feature
        self.local_conv = nn.Conv2d(classifier_input_dim, local_conv_out_channels, 1)
        self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.local_relu = nn.ReLU()

        if use_cuda:
            self.extractor = self.extractor.cuda()
            self.classifier = self.classifier.cuda()
            self.local_conv = self.local_conv.cuda()
            self.local_bn = self.local_bn.cuda()
            self.local_relu = self.local_relu.cuda()
#             self.enc_mu = self.enc_mu.cuda()
# #             self.enc_logvar = self.enc_logvar.cuda()
#             self.encode_mean_logvar = self.encode_mean_logvar.cuda()
    
 
    def forward(self, data):

        features = self.extractor(data=data)

        extracted_feature = features[-1]
        
        latent_feature = self.avgpool(extracted_feature)

        cls_vector = self.classifier(data=latent_feature)
        
        # shape [N, C]
        global_feat = latent_feature.view(latent_feature.size(0), -1)
        # shape [N, C, H, 1]
        local_feat = torch.mean(extracted_feature, -1, keepdim=True)
        local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        # shape [N, H, c]
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

        return_dict = {'latent_vector': latent_feature, 
                       'cls_vector': cls_vector, 
                       'global_feature': global_feat, 
                       'local_feature': local_feat}

        return return_dict
    
    
    
if __name__ == '__main__':
    data = Variable(torch.rand(4,3,256,128)).cuda()
    
    '''
    extractor = Extractor(backbone='resnet-101')
    feature = extractor(data)
    print('extractor output:', len(feature))
    
    cls = Classifier()
    vec = cls(feature[-1])
    print('classifier output:', vec.size())

    decoder = Decoder(backbone='resnet-101')
    reconstruct = decoder(features=feature)
    print('decoder output:', reconstruct.size())
    '''

    extractor = Extractor(backbone='resnet-101')
    print(extractor)
    exit(-1)
    model = AdaptReID()
    f, cls, rec = model(data)
    for idx in range(len(f)):
        print('feature size:', f[idx].size())
    #print('classifier output size:', cls.size())
    #print('reconstruction output size:', rec.size())
