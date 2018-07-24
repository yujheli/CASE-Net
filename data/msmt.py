from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import random
import config
from transform.transform import GeometricTnf

class MSMT(Dataset):
    """
        mode:
            source: 751 (train) + 750 (test) different identities.
            train:  751 different identities.
            test:   750 different identities.
        
        label:
            source:   0 ~ 1500
            train:    0 ~  750
            test:   751 ~ 1500
    """
    
    def __init__(self, 
                 mode='source',
                 dataset_path=config.MSMT_DATA_DIR,
                 csv_path=config.MSMT_CSV_DIR,
                 image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
                 transform=None,
                 random_crop=False): 

        self.image_height, self.image_width = image_size
        self.dataset_path = dataset_path 
        self.csv_path = csv_path 
        self.mode = mode
        self.csv = self.get_csv(mode)
        self.image_names = self.csv['image_path']
        self.image_labels = self.csv['id']
        self.downsample_scale = self.csv['downsample'].as_matrix().astype('int')
        if self.mode == 'test' or self.mode == 'query':
            self.camera_id = self.csv['camera'].as_matrix().astype('int')
        self.random_crop = random_crop
        self.transform = transform
        self.affineTnf = GeometricTnf(out_h=self.image_height, # Use for image resizing
                                      out_w=self.image_width, 
                                      use_cuda=False) 
 
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        input_image, rec_image = self.get_image(self.image_names, idx)
        label = self.get_label(self.image_labels, idx)

        database = {'image': input_image} 

        if self.mode == 'source':
            database['label'] = label
            database['rec_image'] = rec_image

        elif self.mode == 'train':
            database['rec_image'] = rec_image

        elif self.mode == 'test' or self.mode == 'query':
            camera_id = self.camera_id[idx]
            database['camera_id'] = camera_id
            database['label'] = label

        if self.transform:
            database = self.transform(database)
        
        return database

    def get_csv(self, mode):
        if mode == 'source':
            csv_path = os.path.join(self.csv_path, config.SOURCE_DATA_CSV)
        elif mode == 'train':
            csv_path = os.path.join(self.csv_path, config.TRAIN_DATA_CSV)
        elif mode == 'test':
            csv_path = os.path.join(self.csv_path, config.TEST_DATA_CSV)
        else: # query
            csv_path = os.path.join(self.csv_path, config.QUERY_DATA_CSV)

        return pd.read_csv(csv_path)

    def get_image(self, image_list, idx):
        image_name = os.path.join(self.dataset_path, image_list[idx])
        image = io.imread(image_name)

        """ Random Cropping """
        if self.random_crop:
            h,w,c = image.shape
            top = np.random.randint(h/4)
            bottom = int(3*h/4+np.random.randint(h/4))
            left = np.random.randint(w/4)
            right = int(3*w/4+np.random.randint(w/4))
            image = image[top:bottom, left:right, :]
 
        if self.mode == 'train' or self.mode == 'source':
            """ Flip Image """
            if random.random() > 0.5:
                image = np.flip(image, 1) 

        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))

        downsampled_image = self.downsample(image, self.downsample_scale[idx])
        downsampled_image_var = Variable(downsampled_image, requires_grad=False)
        downsampled_image = self.affineTnf(downsampled_image_var).data.squeeze(0)

        image_var = Variable(image, requires_grad=False)
        image = self.affineTnf(image_var).data.squeeze(0)
        
        return downsampled_image, image

    def downsample(self, image, downsample_scale=2):
        kernel = (downsample_scale, downsample_scale)
        downsampled_image = F.max_pool2d(image, kernel)
        return downsampled_image
 
    def get_label(self, label_list, idx):
        label = label_list[idx]
        label = Variable(torch.from_numpy(np.array(label, dtype='int32')).long())
        return label
