from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
import random

class PersonDataset(Dataset):
    
    """
    
    Proposal Flow image pair dataset
    
    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, csv_file, dataset_path, output_size=(224,224), transform=None):

        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.names = self.pairs.iloc[:,0]
        self.labels = self.pairs.iloc[:,1]
        self.dataset_path = dataset_path         
        # self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        # self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
              
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        image = self.get_image(self.names,idx)
        label = self.get_label(self.labels,idx)

        # # get pre-processed point coords
        # point_A_coords = self.get_points(self.point_A_coords,idx)
        # point_B_coords = self.get_points(self.point_B_coords,idx)
        
        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        # L_pck = torch.FloatTensor([torch.max(point_A_coords.max(1)[0]-point_A_coords.min(1)[0])])
        if random.random() > 0.5:
            image = np.fliplr(image)      
        sample = {'image': image, 'label': label}

        return sample

    def get_image(self,img_name_list,idx):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = io.imread(img_name)
        
        # get image size
        im_size = np.asarray(image.shape)
        
        # convert to torch Variable
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        
        
        return image
    
    def get_label(self,point_coords_list,idx):
        point_coords = point_coords_list[idx]

#        # swap X,Y coords, as the the row,col order (Y,X) is used for computations
#        point_coords = point_coords[[1,0],:]

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords