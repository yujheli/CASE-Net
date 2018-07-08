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

class Duke(Dataset):
    
    def __init__(self, 
                 csv_file, 
                 dataset_path, 
                 image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)): 

        self.image_height, self.image_width = image_size
        self.csv = pd.read_csv(csv_file)
        self.image_names = self.csv.iloc[:,0]
        self.image_labels = self.csv.iloc[:,1]
        self.dataset_path = dataset_path 
 
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        image = self.get_image(self.image_names, idx)
        label = self.get_label(self.image_labels,idx)

        if random.random() > 0.5:
            image = np.fliplr(image) 

        database = {'image': image, 
                    'label': label}

        return database

    def get_image(self, image_list, idx):
        image_name = os.path.join(self.dataset_path, image_list[idx])
        image = io.imread(image_name)
        
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))

        return image
 
    def get_label(self, label_list, idx):
        label = label_list[idx]
        label = torch.Tensor(label.astype(np.float32))
        return label
