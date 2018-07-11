from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
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
            self.skip_idx = ['2', '4', '5', '6', '7', '8']
        
        elif backbone == 'resnet-101':
            self.model = models.resnet101(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.skip_idx = ['2', '4', '5', '6', '7', '8']

    def forward(self, data):
        skip_features = []
        for idx, module in self.model._modules.items():
            data = module(data)
            if idx in self.skip_idx:
                skip_features.append(data)
        
        if self.skip_connection:
            return skip_features
        else:
            return skip_features[-2:]

class Classifier(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 output_dim=config.CLASS_NUM):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        data = data.view(data.size()[0],-1)
        out = self.linear(data)
        features = self.softmax(out)
        return features

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
            nn.Sigmoid()
        )

    def forward(self, features):
        if self.skip_connection:
            f1, f2, f3, f4, f5, _ = features
            block1 = self.block1(f5)
            block2 = self.block2(block1+f4)
            block3 = self.block3(block2+f3)
            block4 = self.block4(block3+f2)
            block5 = self.block5(block4+f1)
            return block5
        else:
            f5, _ = features
            block1 = self.block1(f5)
            block2 = self.block2(block1)
            block3 = self.block3(block2)
            block4 = self.block4(block3)
            block5 = self.block5(block4)
            return block5

class AdaptReID(nn.Module):
    def __init__(self,
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION,
                 classifier_input_dim=2048,
                 classifier_output_dim=1812,
                 use_cuda=True):
        super(AdaptReID, self).__init__()

        self.extractor = Extractor(backbone=backbone)

        self.decoder = Decoder(backbone=backbone,
                               skip_connection=skip_connection)

        self.classifier = Classifier(input_dim=classifier_input_dim,
                                     output_dim=classifier_output_dim)

        self.skip_connection = skip_connection

        if use_cuda:
            self.extractor = self.extractor.cuda()
            self.decoder = self.decoder.cuda()
            self.classifier = self.classifier.cuda()

    def forward(self, data):

        features = self.extractor(data=data)

        latent_feature = features[-1]
        extracted_feature = features[-2]

        cls_vector = self.classifier(data=latent_feature)

        reconstruct = self.decoder(features=features)

        return latent_feature, extracted_feature, cls_vector, reconstruct

 
if __name__ == '__main__':
    data = Variable(torch.rand(4,3,224,224)).cuda()
    
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
