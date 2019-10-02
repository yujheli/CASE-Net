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
from torchvision import transforms


class VERI(Dataset):
    """
        mode:
            source: 576 (train) + 200(test) different identities.
            train:  576 different identities.
            test:   200 different identities.
        
        label:
            source:   0 ~ 776
            train:    0 ~ 576
            test:     0 ~ 200
    """
    
    def __init__(self, 
                 mode='source',
                 dataset_path=config.Veri_DATA_DIR,
                 csv_path=config.Veri_CSV_DIR,
                 image_size=(config.Vi_IMAGE_HEIGHT, config.Vi_IMAGE_WIDTH),
                 transform=None,
                 random_crop=False,
                 im_per_id=4, ds_factor=4): 

        self.image_height, self.image_width = image_size
        self.mode = mode
        
        if self.mode == 'test':
            self.dataset_path = os.path.join(dataset_path, 'image_test')
            self.csv_path = os.path.join(csv_path, 'veri_test_list.txt')
            
        elif self.mode == 'query':
            self.dataset_path = os.path.join(dataset_path, 'image_query')
            self.csv_path = os.path.join(csv_path, 'veri_query_list.txt')
            
        elif self.mode == 'train' or self.mode == 'source':
            self.dataset_path = os.path.join(dataset_path, 'image_train')
            self.csv_path = os.path.join(csv_path, 'veri_train_list.txt')
            
        else:
            print("Error setting data loading path")
            
            
        
        
        
#         self.csv = self.get_csv(mode)
#         self.image_names = self.csv['image_path']
#         self.image_labels = self.csv['id']
#         self.downsample_scale = self.csv['downsample'].as_matrix().astype('int')

        self.downsample_scale = ds_factor
    
        self.image_names = []
        self.image_labels = []
        self.camera_id = []
        
        reader = open(self.csv_path)
        lines = reader.readlines()
                
        if self.mode == 'test' or self.mode == 'query':
            for line in lines:
                line = line.strip()
                self.image_names.append(line)
                self.image_labels.append(int(line.split('_')[0]))
                self.camera_id.append(int(line.split('_')[1][1:]))
        else:
            for line in lines:
                line = line.strip().split(' ')
                self.image_names.append(line[0])
                self.image_labels.append(int(line[1]))
                self.camera_id.append(int(line[0].strip().split('_')[1][1:]))
        
#         if self.mode == 'test' or self.mode == 'query':
#             self.camera_id = self.csv['camera'].as_matrix().astype('int')
        self.random_crop = random_crop
        self.transform = transform
        self.affineTnf = GeometricTnf(out_h=self.image_height, # Use for image resizing
                                      out_w=self.image_width, 
                                      use_cuda=False)
        self.im_per_id = im_per_id

        self.hash_table = self.set_dict()

        self.normalize = transforms.Normalize(
            mean=config.MEAN,
            std=config.STDDEV
        )
    
    def set_dict(self):
        hash_table = dict()
        for idx in range(len(self.image_names)):
            label = int(self.image_labels[idx])
            if label not in hash_table:
                hash_table[label] = []
                hash_table[label].append(idx)
            else:
                hash_table[label].append(idx)
        return hash_table
 
    def __len__(self):
        if self.mode == 'test' or self.mode == 'query':
            return len(self.image_names)
        else:
            return len(self.hash_table)


    def __getitem__(self, idx):
        
        if self.mode == 'test' or self.mode == 'query':
            input_image, rec_image = self.get_image(self.image_names, idx)
            label = self.get_label(self.image_labels, idx)

            database = {'image': input_image} 

            camera_id = self.camera_id[idx]
            database['camera_id'] = camera_id
            database['label'] = label

            if self.transform:
                database = self.transform(database)

            return database

        else:
            inds = self.hash_table[idx]
            if len(inds) < self.im_per_id:
                inds = np.random.choice(inds, self.im_per_id, replace=False)
            else:
                inds = np.random.choice(inds, self.im_per_id, replace=False)

            input_image_list = []
            input_rec_list = []
            for id_ in inds:
                input_image, rec_image = self.get_image(self.image_names, id_)
                input_image_list.append(self.normalize(input_image/255.0).unsqueeze(0))
                input_rec_list.append(self.normalize(rec_image/255.0).unsqueeze(0))

            input_image_list = torch.cat(input_image_list)
            input_rec_list = torch.cat(input_rec_list)

            label = Variable(torch.from_numpy(np.array([idx] * self.im_per_id, dtype='int32')).long())

            database = {'image': input_image_list} 

#             if self.mode == 'source':
#                 database['label'] = label
#                 database['rec_image'] = input_rec_list #rec_image

#             elif self.mode == 'train':
#                 database['label'] = label
#                 database['rec_image'] = input_rec_list #rec_image
                
            database['label'] = label
            database['rec_image'] = input_rec_list #rec_image

            return database

    def get_csv(self, mode):
        if mode == 'source':
            csv_path = os.path.join(self.csv_path, config.SOURCE_DATA_CSV)
        elif mode == 'train':
            csv_path = os.path.join(self.csv_path, config.TRAIN_DATA_CSV)
        elif mode == 'semi':
            csv_path = os.path.join(self.csv_path, config.SEMI_DATA_CSV)
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
            top = np.random.randint(h/8)
            bottom = int(7*h/8+np.random.randint(h/8))
            left = np.random.randint(w/8)
            right = int(7*w/8+np.random.randint(w/8))
            image = image[top:bottom, left:right, :]
 
        if self.mode == 'train' or self.mode == 'source' or self.mode == 'semi':
            """ Flip Image """
            if random.random() > 0.5:
                image = np.flip(image, 1) 

        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))

#         downsampled_image = self.downsample(image, self.downsample_scale[idx])
        downsampled_image = self.downsample(image, self.downsample_scale)

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
