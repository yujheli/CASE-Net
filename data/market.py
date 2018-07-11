from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import random
import config
from transform.transform import GeometricTnf

class Market(Dataset):
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
                 dataset_path=config.MARKET_DIR,
                 image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
                 transform=None,
                 random_crop=False): 

        self.image_height, self.image_width = image_size
        self.dataset_path = dataset_path 
        self.csv = self.get_csv(mode)
        self.image_names = self.csv.iloc[:,0]
        self.image_labels = self.csv.iloc[:,1]
        self.random_crop = random_crop
        self.transform = transform
        self.affineTnf = GeometricTnf(out_h=self.image_height, 
                                      out_w=self.image_width, 
                                      use_cuda=False) 
 
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        image = self.get_image(self.image_names, idx)
        label = self.get_label(self.image_labels, idx)

        database = {'image': image, 
                    'label': label}

        if self.transform:
            database = self.transform(database)
        
        return database

    def get_csv(self, mode):
        if mode == 'source':
            csv_path = os.path.join(self.dataset_path, 'all_list.csv')
            self.class_num = 1501

        elif mode == 'train':
            csv_path = os.path.join(self.dataset_path, 'train_list.csv')
            self.class_num = 751

        else:
            csv_path = os.path.join(self.dataset_path, 'test_list.csv')
            self.class_num = 750

        return pd.read_csv(csv_path)

    def get_image(self, image_list, idx):
        image_name = os.path.join(self.dataset_path, image_list[idx])
        image = io.imread(image_name)

        if self.random_crop:
            h,w,c = image.shape
            top = np.random.randint(h/4)
            bottom = int(3*h/4+np.random.randint(h/4))
            left = np.random.randint(w/4)
            right = int(3*w/4+np.random.randint(w/4))
            image = image[top:bottom, left:right, :]
 
        if random.random() > 0.5:
            image = np.flip(image, 1) 

        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)
        image = self.affineTnf(image_var).data.squeeze(0)
        
        return image
 
    def get_label(self, label_list, idx):
        label = label_list[idx]
        label = Variable(torch.from_numpy(np.array(label, dtype='int32')).long())
        return label
