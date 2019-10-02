from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import config

class Extractor(nn.Module):
    def __init__(self,
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION):
        super(Extractor, self).__init__()
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

    def forward(self, data):
        skip_features = []
        for idx, module in self.model._modules.items():
            data = module(data)
            if idx in self.skip_idx:
                skip_features.append(data)
        
        return skip_features

class Classifier(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 output_dim=config.DUKE_CLASS_NUM):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        data = data.view(data.size()[0],-1)
        out = self.linear(data)
        return out
    
    


class Decoder(nn.Module):
    def __init__(self,
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION):
        super(Decoder, self).__init__()

        self.skip_connection = skip_connection

        if backbone == 'resnet-50' or backbone == 'resnet-101':
            channel_list = [2048, 1024, 512, 256, 64, 3]

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[0], channel_list[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[1], channel_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[1], channel_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[1], channel_list[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[2], channel_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[2], channel_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[2], channel_list[3], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[3], channel_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[3], channel_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[3], channel_list[4], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[4]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[4], channel_list[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[4]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[4], channel_list[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[4]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[4], channel_list[5], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[5]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[5], channel_list[5], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[5]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[5], channel_list[5], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[5]),
            nn.Tanh()
        )

    def forward(self, features):
        if self.skip_connection:
            f1, f2, f3, f4, f5 = features
            block1 = self.block1(f5)
            block2 = self.block2(block1+f4)
            block3 = self.block3(block2+f3)
            block4 = self.block4(block3+f2)
            block5 = self.block5(block4+f1)
            return block5
        else:
            f1, f2, f3, f4, f5 = features
            block1 = self.block1(f5)
            block2 = self.block2(block1+f4)
            block3 = self.block3(block2)
            block4 = self.block4(block3)
            block5 = self.block5(block4)
            return block5

  
        
        
        
class AdaptReID(nn.Module):
    def __init__(self,
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION,
                 classifier_input_dim=2048,
                 classifier_output_dim=config.DUKE_CLASS_NUM,
                 use_cuda=True,
                 local_conv_out_channels=128,
                 ):
        super(AdaptReID, self).__init__()

        self.extractor = Extractor(backbone=backbone)

        self.decoder = Decoder(backbone=backbone,
                               skip_connection=skip_connection)

        self.classifier = Classifier(input_dim=classifier_input_dim,
                                     output_dim=classifier_output_dim)

        self.skip_connection = skip_connection

        self.avgpool = nn.AvgPool2d((8,4))
        
        #local feature
        self.local_conv = nn.Conv2d(classifier_input_dim, local_conv_out_channels, 1)
        self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.local_relu = nn.ReLU(inplace=True)

        if use_cuda:
            self.extractor = self.extractor.cuda()
            self.decoder = self.decoder.cuda()
            self.classifier = self.classifier.cuda()
            self.local_conv = self.local_conv.cuda()
            self.local_bn = self.local_bn.cuda()
            self.local_relu = self.local_relu.cuda()
 
    def forward(self, data):

        features = self.extractor(data=data)

        extracted_feature = features[-1]

        latent_feature = self.avgpool(extracted_feature)

        cls_vector = self.classifier(data=latent_feature)

        reconstruct = self.decoder(features=features)
        
        # shape [N, C]
        global_feat = latent_feature.view(latent_feature.size(0), -1)
        # shape [N, C, H, 1]
        local_feat = torch.mean(extracted_feature, -1, keepdim=True)
        local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        # shape [N, H, c]
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

        return latent_feature, features, cls_vector, reconstruct, global_feat, local_feat

class Encode_Mean_Logvar(nn.Module):
    def __init__(self, input_channel = 2048, output_channel = 4096):
        super(Encode_Mean_Logvar, self).__init__()
        self.enc_mu = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2,padding=1)
        self.enc_logvar = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2,padding=1)
        
    def forward(self, data):
        mu = self.enc_mu(data)
        logvar = self.enc_logvar(data)

        return mu, logvar
    
class VAE_Decoder(nn.Module):
    def __init__(self,
                 backbone='resnet-101', input_dim=4096, code_dim=750):
        super(VAE_Decoder, self).__init__()

#         self.skip_connection = skip_connection
        self.input_dim = input_dim
        self.code_dim = code_dim

        if backbone == 'resnet-50' or backbone == 'resnet-101':
            channel_list = [2048, 1024, 512, 256, 64, 3]
            
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim+self.code_dim, channel_list[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(channel_list[0], channel_list[1], kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel_list[1]),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(channel_list[0], channel_list[1], kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel_list[1]),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
            
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[0], channel_list[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[1], channel_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[1], channel_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[1], channel_list[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[2], channel_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[2], channel_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[2], channel_list[3], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[3], channel_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[3], channel_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[3], channel_list[4], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[4]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[4], channel_list[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[4]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[4], channel_list[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[4]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[4], channel_list[5], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[5]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[5], channel_list[5], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[5]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[5], channel_list[5], kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel_list[5]),
            nn.Tanh()
        )

    def forward(self, data, features=None):
        
        if features is not None:
            f1, f2, f3, f4, f5 = features
            block0 = self.block0(data)
            block1 = self.block1(block0+f5)
            block2 = self.block2(block1+f4)
            block3 = self.block3(block2)
            block4 = self.block4(block3)
            block5 = self.block5(block4)
        else:
            block0 = self.block0(data)
            block1 = self.block1(block0)
            block2 = self.block2(block1)
            block3 = self.block3(block2)
            block4 = self.block4(block3)
            block5 = self.block5(block4)
        
        return block5      

    
class Res_Decoder(nn.Module):
    def __init__(self,
                 backbone='resnet-101', input_dim=4096, code_dim=750, skip_connection=False):
        super(Res_Decoder, self).__init__()

        self.skip_connection = skip_connection
        self.input_dim = input_dim
        self.code_dim = code_dim

        
        channel_list = [1024, 512, 256, 128, 64, 3]
        if self.skip_connection:
#             skipped_list = [0, 0, 512, 256, 0] # skip f3, f2
            skipped_list = [0, 1024, 512, 256, 0] # skip f4, f3, f2
            #skipped_list = [2048, 1024, 512, 256, 64] # skip f3, f2
            full_skipped_list = [2048, 1024, 512, 256, 64]
        else:
            skipped_list = [0, 0, 0, 0, 0]

        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim+self.code_dim, channel_list[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], channel_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[0])
#             nn.ReLU(inplace=True),
        )
        self.residue_0 = nn.Sequential(
#             nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.input_dim+self.code_dim, channel_list[0], kernel_size=1, stride=2, output_padding=1),
            nn.BatchNorm2d(channel_list[0])
            )
            
        self.block1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel_list[0] + skipped_list[0], channel_list[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.ReLU(inplace=True),
            #droupout
            nn.Conv2d(channel_list[1], channel_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[1])
            
        )
        self.residue_1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(channel_list[0], channel_list[1], kernel_size=1, stride=2, output_padding=1),
            nn.BatchNorm2d(channel_list[1])
            )

        self.block2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel_list[1]+ skipped_list[1], channel_list[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.ReLU(inplace=True),
           
            nn.Conv2d(channel_list[2], channel_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[2])
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel_list[2], channel_list[2], kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel_list[2]),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.residue_2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(channel_list[1], channel_list[2], kernel_size=1, stride=2,output_padding=1),
            nn.BatchNorm2d(channel_list[2])
            )

        self.block3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel_list[2] + skipped_list[2], channel_list[3], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[3], channel_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[3])
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel_list[3], channel_list[3], kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel_list[3]),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.residue_3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(channel_list[2], channel_list[3], kernel_size=1, stride=2, output_padding=1),
            nn.BatchNorm2d(channel_list[3])
            )

        self.block4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel_list[3] + skipped_list[3], channel_list[4], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[4], channel_list[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[4])
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel_list[4], channel_list[4], kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel_list[4]),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.residue_4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(channel_list[3], channel_list[4], kernel_size=1, stride=2, output_padding=1),
            nn.BatchNorm2d(channel_list[4])
            )

        self.block5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel_list[4] + skipped_list[4], channel_list[5], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel_list[5]),
            nn.ReLU(inplace=True),
#             nn.Conv2d(channel_list[5], channel_list[5], kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel_list[5]),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_list[5], channel_list[5], kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel_list[5]),
            nn.Tanh()
        )
        
        
        if self.skip_connection:
            self.f2_block = nn.Sequential(
            nn.Conv2d(full_skipped_list[4], full_skipped_list[3], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(full_skipped_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(full_skipped_list[3], full_skipped_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(full_skipped_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
       
            )
            
            self.f3_block = nn.Sequential(
            nn.Conv2d(full_skipped_list[3], full_skipped_list[2], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(full_skipped_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(full_skipped_list[2], full_skipped_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(full_skipped_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            
            self.f4_block = nn.Sequential(
            nn.Conv2d(full_skipped_list[2], full_skipped_list[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(full_skipped_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(full_skipped_list[1], full_skipped_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(full_skipped_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            
            

        
    def forward(self, data, features=None):
        
        if features is not None:
            f1, f2, f3, f4, f5 = features
            
            f2_e = self.f2_block(f1)
            f3_e = self.f3_block(f2)
            f4_e = self.f4_block(f3)
            
            skip_features_e = [f2_e, f3_e, f4_e]
#             block0 = self.block0(data)
#             block1 = self.block1(block0+f5)
#             block2 = self.block2(block1+f4)
#             block3 = self.block3(block2)
#             block4 = self.block4(block3)
#             block5 = self.block5(block4)
            
            block0 = self.block0(data)
            residue0 = self.residue_0(data)
            out0 = block0+residue0
            
            block1 = self.block1(out0)
            residue1 = self.residue_1(out0)
            out1 = block1+residue1
            
            block2 = self.block2(torch.cat([out1, f4_e],dim=1))
            residue2 = self.residue_2(out1)
            out2 = block2+residue2
            
            block3 = self.block3(torch.cat([out2, f3_e],dim=1))
            residue3 = self.residue_3(out2)
            out3 = block3+residue3
            
#             block4 = self.block4(out3)
            block4 = self.block4(torch.cat([out3,f2_e],dim=1))
            residue4 = self.residue_4(out3)
            out4 = block4+residue4
            
            block5 = self.block5(out4)
            
            return block5, skip_features_e
            
        else:
            block0 = self.block0(data)
            residue0 = self.residue_0(data)
            out0 = block0+residue0
            
            block1 = self.block1(out0)
            residue1 = self.residue_1(out0)
            out1 = block1+residue1
            
            block2 = self.block2(out1)
            residue2 = self.residue_2(out1)
            out2 = block2+residue2
            
            block3 = self.block3(out2)
            residue3 = self.residue_3(out2)
            out3 = block3+residue3
            
            block4 = self.block4(out3)
            residue4 = self.residue_4(out3)
            out4 = block4+residue4
            
            block5 = self.block5(out4)
        
            return block5    
    
class AdaptVAEReID(nn.Module):
    def __init__(self,
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION,
                 classifier_input_dim=2048,
                 classifier_output_dim=config.DUKE_CLASS_NUM,
                 use_cuda=True,
                 local_conv_out_channels=128,mu_dim=2048, code_dim=config.DUKE_CLASS_NUM):
        super(AdaptVAEReID, self).__init__()

        self.extractor = Extractor(backbone=backbone)
        
        self.code_dim = code_dim
        self.mu_dim = mu_dim
        self.skip_connection = skip_connection
        
#         self.enc_mu = nn.Conv2d(classifier_input_dim, self.mu_dim, kernel_size=4, stride=2,padding=1)
#         self.enc_logvar = nn.Conv2d(classifier_input_dim, self.mu_dim, kernel_size=4, stride=2,padding=1)
        self.encode_mean_logvar = Encode_Mean_Logvar(input_channel = classifier_input_dim, output_channel = self.mu_dim)

        self.decoder = Res_Decoder(backbone=backbone, input_dim=self.mu_dim, code_dim=self.code_dim, skip_connection=self.skip_connection)

        self.classifier = Classifier(input_dim=classifier_input_dim,
                                     output_dim=classifier_output_dim)

        

        self.avgpool = nn.AvgPool2d((8,4))
        
        #local feature
        self.local_conv = nn.Conv2d(classifier_input_dim, local_conv_out_channels, 1)
        self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.local_relu = nn.ReLU()

        if use_cuda:
            self.extractor = self.extractor.cuda()
            self.decoder = self.decoder.cuda()
            self.classifier = self.classifier.cuda()
            self.local_conv = self.local_conv.cuda()
            self.local_bn = self.local_bn.cuda()
            self.local_relu = self.local_relu.cuda()
#             self.enc_mu = self.enc_mu.cuda()
#             self.enc_logvar = self.enc_logvar.cuda()
            self.encode_mean_logvar = self.encode_mean_logvar.cuda()
            
            
    
    def decode(self, z, insert_attrs = None, features=None):
        
        if len(z.size()) != 4:
            z = z.view(z.size()[0],self.mu_dim,4,2)

        if insert_attrs is not None:
            if len(z.size()) == 2:
                z = torch.cat([z,insert_attrs],dim=1)
            else:
                H,W = z.size()[2], z.size()[3]
                z = torch.cat([z,insert_attrs.unsqueeze(-1).unsqueeze(-1).repeat(1,1,H,W)],dim=1)
#                 print(z.size())
        reconstruct, skip_features_e = self.decoder(data=z, features=features)
       
        return reconstruct, skip_features_e
    
    def encode(self, x):
#         for l in range(len(self.enc_layers)-1):
#             if (self.enc_layers[l] == 'fc')  and (len(x.size())>2):
#                 batch_size = x.size()[0]
#                 x = x.view(batch_size,-1)
#             x = getattr(self, 'enc_'+str(l))(x)

#         if (self.enc_layers[-1] == 'fc')  and (len(x.size())>2):
#             batch_size = x.size()[0]
#             x = x.view(batch_size,-1)
        features = self.extractor(data=x)
        extracted_feature = features[-1]
#         mu = self.enc_mu(extracted_feature)
#         logvar = self.enc_logvar(extracted_feature)
        mu, logvar =  self.encode_mean_logvar(extracted_feature)
        if len(mu.size()) > 2:
            mu = mu.view(mu.size()[0],-1)
            logvar = logvar.view(mu.size()[0],-1)

        return mu, logvar
    

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
 
    def forward(self, data, insert_attrs=None, return_enc=False):

        features = self.extractor(data=data)

        extracted_feature = features[-1]
        
        mu, logvar =  self.encode_mean_logvar(extracted_feature)
#         mu = self.enc_mu(extracted_feature)
#         logvar = self.enc_logvar(extracted_feature)

        latent_feature = self.avgpool(extracted_feature)

        cls_vector = self.classifier(data=latent_feature)
        
        
        
        if len(mu.size()) > 2:
            mu = mu.view(mu.size()[0],-1)
            logvar = logvar.view(mu.size()[0],-1)
        z = self.reparameterize(mu, logvar)
#         z = Variable(mu.data.new(mu.size()).normal_())

#         reconstruct = self.decoder(features=features)

        if self.skip_connection:
            skip_features = features
        else:
            skip_features = None
    
        if insert_attrs is not None:
            reconstruct, skip_features_e = self.decode(z, insert_attrs, features=skip_features)    
        else:
            reconstruct = self.decode(z, cls_vector, features=skip_features)
            

        
        # shape [N, C]
        global_feat = latent_feature.view(latent_feature.size(0), -1)
        # shape [N, C, H, 1]
        local_feat = torch.mean(extracted_feature, -1, keepdim=True)
        local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        # shape [N, H, c]
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

        #return latent_feature, features, cls_vector, reconstruct, global_feat, local_feat, mu, logvar
        #return {'latent_vector': latent_feature, 

        resolution_feature = features[-1]

        return_dict = {'latent_vector': latent_feature, 
                       'resolution_feature': resolution_feature, 
                       'cls_vector': cls_vector, 
                       'rec_image': reconstruct, 
                       'global_feature': global_feat, 
                       'local_feature': local_feat, 
                       'mu': mu, 
                       'logvar': logvar,
                       'image': data,
                       'features': features[1:4]}

        if insert_attrs is not None:
            return_dict['skip_e'] = skip_features_e
        else:
            return_dict['skip_e'] = cls_vector

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
