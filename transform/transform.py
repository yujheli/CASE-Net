from __future__ import print_function, division
import os
import sys
from skimage import io
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from util.torch_util import expand_dim

class GeometricTnf(object):
    """
    
    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )        
    
    """
    def __init__(self, 
                 geometric_model='affine', 
                 tps_grid_size=3, 
                 tps_reg_factor=0, 
                 out_h=224, 
                 out_w=224, 
                 offset_factor=None, 
                 use_cuda=True):

        self.out_h = out_h
        self.out_w = out_w
        self.geometric_model = geometric_model
        self.use_cuda = use_cuda
        self.offset_factor = offset_factor
        
        if geometric_model == 'affine' and offset_factor is None:
            self.gridGen = AffineGridGen(out_h=out_h, out_w=out_w, use_cuda=use_cuda)

        elif geometric_model == 'affine' and offset_factor is not None:
            self.gridGen = AffineGridGenV2(out_h=out_h, out_w=out_w, use_cuda=use_cuda)

        elif geometric_model == 'tps':
            self.gridGen = TpsGridGen(out_h=out_h,
                                      out_w=out_w, 
                                      grid_size=tps_grid_size, 
                                      reg_factor=tps_reg_factor, 
                                      use_cuda=use_cuda)
        if offset_factor is not None:
            self.gridGen.grid_X = self.gridGen.grid_X / offset_factor
            self.gridGen.grid_Y = self.gridGen.grid_Y / offset_factor   
            
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1,0,0],[0,1,0]]), 0).astype(np.float32))

        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, 
                 image_batch, 
                 theta_batch=None, 
                 out_h=None, 
                 out_w=None, 
                 return_warped_image=True, 
                 return_sampling_grid=False, 
                 padding_factor=1.0, 
                 crop_factor=1.0):

        if image_batch is None:
            b = 1

        else:
            b = image_batch.size(0)

        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b,2,3).contiguous()
            theta_batch = Variable(theta_batch,requires_grad=False)        
        
        # check if output dimensions have been specified at call time and have changed
        if (out_h is not None and out_w is not None) and (out_h!=self.out_h or out_w!=self.out_w):
            if self.geometric_model == 'affine':
                gridGen = AffineGridGen(out_h, out_w)
            elif self.geometric_model == 'tps':
                gridGen = TpsGridGen(out_h, out_w, use_cuda=self.use_cuda)
        else:
            gridGen = self.gridGen
        
        sampling_grid = gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        if padding_factor != 1 or crop_factor !=1:
            sampling_grid = sampling_grid*(padding_factor*crop_factor)
        # rescale grid according to offset_factor
        if self.offset_factor is not None:
            sampling_grid = sampling_grid*self.offset_factor
        
        if return_sampling_grid and not return_warped_image:
            return sampling_grid
        
        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid)
        
        if return_sampling_grid and return_warped_image:
            return (warped_image_batch,sampling_grid)
        
        return warped_image_batch

class AffineGridGen(Module):
    def __init__(self, out_h=224, out_w=224, out_ch=3, use_cuda=True):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        b=theta.size()[0]
        if not theta.size() == (b,2,3):
            theta = theta.view(-1,2,3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size)
