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
                 skip_connection=config.SKIP_CONNECTION,
                 use_cuda=True):
        super(Extractor, self).__init__()
        self.skip_connection = skip_connection
        if backbone == 'resnet-50':
            self.model = models.resnet50(pretrained=True)
            self.skip_idx = ['2', '4', '5', '6', '7']
        
        elif backbone == 'resnet-101':
            self.model = models.resnet101(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.skip_idx = ['2', '4', '5', '6', '7']

        if use_cuda:
            self.model = self.model.cuda()
    
    def forward(self, data):
        skip_features = []
        for idx, module in self.model._modules.items():
            data = module(data)
            if idx in self.skip_idx:
                print(data.size())
                skip_features.append(data)
        
        if self.skip_connection:
            return skip_features
        else:
            return skip_features[-1]

class Classifier(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 output_dim=config.CLASS_NUM,
                 use_cuda=True):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        if use_cuda:
            self.linear = self.linear.cuda()

    def forward(self, data):
        data = data.view(data.size()[0],-1)
        features = self.linear(data)
        return features

class Decoder(nn.Module):
    def __init__(self,
                 backbone='resnet-101',
                 skip_connection=config.SKIP_CONNECTION):
        super(Decoder, self).__init__()
        if backbone == 'resnet-50' or backbone == 'resnet-101':
            channel_list = [2048, 1024, 512, 256, 64, 3]

        self.block1 = nn.Sequential(
            self.conv(channel_list[0], channel_list[1], stride=2, transpose=True),
            self.bn(channel_list[1]),
            nn.LeakyReLU(),
            self.conv(channel_list[1], channel_list[1]),
            self.bn(channel_list[1]),
            nn.LeakyReLU(),
            self.conv(channel_list[1], channel_list[1]),
            self.bn(channel_list[1]),
            nn.LeakyReLU()
        )

        self.block2 = nn.Sequential(
            self.conv(channel_list[1], channel_list[2], stride=2, transpose=True),
            self.bn(channel_list[2]),
            nn.LeakyReLU(),
            self.conv(channel_list[2], channel_list[2]),
            self.bn(channel_list[2]),
            nn.LeakyReLU(),
            self.conv(channel_list[2], channel_list[2]),
            self.bn(channel_list[2]),
            nn.LeakyReLU()
        )

        self.block3 = nn.Sequential(
            self.conv(channel_list[2], channel_list[3], stride=2, transpose=True),
            self.bn(channel_list[3]),
            nn.LeakyReLU(),
            self.conv(channel_list[3], channel_list[3]),
            self.bn(channel_list[3]),
            nn.LeakyReLU(),
            self.conv(channel_list[3], channel_list[3]),
            self.bn(channel_list[3]),
            nn.LeakyReLU()
        )

        self.block4 = nn.Sequential(
            self.conv(channel_list[3], channel_list[4], stride=2, transpose=True),
            self.bn(channel_list[4]),
            nn.LeakyReLU(),
            self.conv(channel_list[4], channel_list[4]),
            self.bn(channel_list[4]),
            nn.LeakyReLU(),
            self.conv(channel_list[4], channel_list[4]),
            self.bn(channel_list[4]),
            nn.LeakyReLU()
        )

        self.block5 = nn.Sequential(
            self.conv(channel_list[4], channel_list[5], stride=2, transpose=True),
            self.bn(channel_list[5]),
            nn.LeakyReLU(),
            self.conv(channel_list[5], channel_list[5]),
            self.bn(channel_list[5]),
            nn.LeakyReLU(),
            self.conv(channel_list[5], channel_list[5]),
            self.bn(channel_list[5]),
            nn.Sigmoid()
        )

    def forward(self, skip_features=None):
        if skip_features is not None:
            f1, f2, f3, f4, feature = skip_feature
            block1 = self.block1(feature)
            block2 = self.block2(block1+f4)
            block3 = self.block3(block2+f3)
            block4 = self.block4(block3+f2)
            block5 = self.block5(block4+f1)
            return block5
        else:
            block1 = self.block1(feature)
            block2 = self.block2(block1)
            block3 = self.block3(block2)
            block4 = self.block4(block3)
            block5 = self.block5(block4)
            return block5

    def bn(self, channel):
        layer = nn.BatchNorm2d(channel)
        nn.init.constant(layer.weight, 1)
        nn.init.constant(layer.bias, 0)
        return layer

    def conv(self, 
             in_channel, 
             out_channel, 
             kernel_size=3, 
             stride=1, 
             dilation=1, 
             bias=False, 
             transpose=False):

        if transpose:
            layer = nn.ConvTranspose2d(in_channel, 
                                       out_channel, 
                                       kernel_size=kernel_size, 
                                       stride=stride, 
                                       padding=1, 
                                       output_padding=1, 
                                       dilation=dilation, 
                                       bias=bias)
            w = torch.Tensor(kernel_size, kernel_size)
            center = kernel_size % 2 == 1 and stride -1 or stride - 0.5
            for y in range(kernel_size):
                for x in range(kernel_size):
                    w[y,x] = (1 - abs((x-center) / stride)) * (1 - abs((y-center) / stride))
            layer.weight.data.copy_(w.div(in_channel).repeat(out_channel, in_channel, 1, 1))

        else:
            padding = (kernel_size + 2*(dilation - 1)) // 2
            layer = nn.Conv2d(in_channel, 
                              out_channel, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              dilation=dilation, 
                              bias=bias)
        if bias:
            nn.init.const(layer.bias, 0)

        return layer

if __name__ == '__main__':
    data = Variable(torch.rand(4,3,224,224)).cuda()
    net = Extractor(backbone='resnet-101')
    feature = net(data)
    print(len(feature))
    #print(feature.size())
    #cls = Classifier()
    #vec = cls(feature)
    #print(vec.size())
