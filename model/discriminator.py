from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self,
                 input_channel=2048,
                 fc_input_dim=1024,
                 fc_output_dim=1,
                 use_cuda=True):
        super(Discriminator, self).__init__()

        channel_list = [input_channel, 1024, 512, 256, 128, 64, 32]

        self.block = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[1], channel_list[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[2], channel_list[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[3], channel_list[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[4]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[4], channel_list[5], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[5]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[5], channel_list[6], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[6]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.linear = nn.Linear(fc_input_dim, fc_output_dim)

        if use_cuda:
            self.block = self.block.cuda()
            self.linear = self.linear.cuda()

    def forward(self, data):
        output = self.block(data)
        output = self.linear(output.view(output.size()[0], -1))
        return output


class ACGAN(nn.Module):
    def __init__(self,
                 input_channel=3,
                 fc_input_dim=1024,
                 class_num=1,
                 use_cuda=True):
        super(ACGAN, self).__init__()

#         channel_list = [input_channel, 1024, 512, 256, 128, 64, 32]
        channel_list = [input_channel, 32, 64, 128, 256, 512, 1024]

        self.block = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[1], channel_list[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[2], channel_list[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[3], channel_list[4], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel_list[4]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[4], channel_list[5], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel_list[5]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(channel_list[5], channel_list[6], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[6]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.linear = nn.Linear(fc_input_dim, 1)
        self.classifier = nn.Linear(fc_input_dim, class_num)
        self.avgpool = nn.AvgPool2d((8,4))

        if use_cuda:
            self.block = self.block.cuda()
            self.linear = self.linear.cuda()
            self.classifier = self.classifier.cuda()
            self.avgpool = self.avgpool.cuda()

    def forward(self, data):
#         print(data.size())
        output = self.block(data)
    
        fc_input = self.avgpool(output)
        real_fake = self.linear(fc_input.view(fc_input.size()[0], -1))
        class_prob = self.classifier(fc_input.view(fc_input.size()[0], -1))
        return real_fake, class_prob
#         return output, class_prob
