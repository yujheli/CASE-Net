from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self,
                 output_dim=2,
                 use_cuda=True):
        super(Discriminator, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 4, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        if use_cuda:
            self.block = self.block.cuda()

    def forward(self, data):
        output = self.block(data)
        return output
