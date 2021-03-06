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
import cv2


def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = recover(x[i,:,:,:])   # 3 channel, 256x128x3
        xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY) # 256x128x1
        xx = cv2.Canny(xx, 10, 200) #256x128
        xx = xx/255.0 - 0.5 # {-0.5,0.5}
        xx += np.random.randn(xx.shape[0],xx.shape[1])*0.1  #add random noise
        xx = torch.from_numpy(xx.astype(np.float32))
        out[i,:,:] = xx
    out = out.unsqueeze(1) 
    return out

def to_gray(x):
    x = x.data.cpu().numpy().transpose((1, 2, 0))
#     inp = inp.numpy().transpose((1, 2, 0))
    x = x.astype(np.uint8)
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    img2 = np.zeros_like(x)
    img2[:,:,0] = gray
    img2[:,:,1] = gray
    img2[:,:,2] = gray
    xx = torch.from_numpy(img2.transpose((2,0,1)).astype(np.float32))
    return xx


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
                 random_crop=True,
                 im_per_id=4, ds_factor=4, q_gray=False, g_gray=False, train_gray_p=0):

        self.image_height, self.image_width = image_size
        self.dataset_path = dataset_path 
        self.csv_path = csv_path 
        self.mode = mode
        self.csv = self.get_csv(mode)
        self.image_names = self.csv['image_path']
        self.image_labels = self.csv['id']
#         self.downsample_scale = self.csv['downsample'].as_matrix().astype('int')
        self.downsample_scale = ds_factor
        if self.mode == 'test' or self.mode == 'query':
            self.camera_id = self.csv['camera'].astype('int')
        self.random_crop = random_crop
        self.transform = transform
        self.affineTnf = GeometricTnf(out_h=self.image_height, # Use for image resizing
                                      out_w=self.image_width, 
                                      use_cuda=False) 
        self.im_per_id = im_per_id
        self.query_gray = q_gray
        self.gallery_gray = g_gray
        self.train_gray_p = train_gray_p
        self.hash_table = self.set_dict()

        self.normalize = transforms.Normalize(
            mean=config.MEAN,
            std=config.STDDEV
        )
        
    def set_dict(self):
        hash_table = dict()
        for idx in range(len(self.csv)):
            label = int(self.image_labels[idx])
            if label not in hash_table:
                hash_table[label] = []
                hash_table[label].append(idx)
            else:
                hash_table[label].append(idx)
        return hash_table
 
 
    def __len__(self):
        if self.mode == 'test' or self.mode == 'query':
            return len(self.csv)
        else:
            return len(self.hash_table)

    def __getitem__(self, idx):
        
        if self.mode == 'test':
            input_image, rec_image = self.get_image(self.image_names, idx)
            label = self.get_label(self.image_labels, idx)
            
            if self.gallery_gray:
                input_image = self.normalize(to_gray(input_image)/255.0)
            else:
                input_image = self.normalize(input_image/255.0)
                
            database = {'image': input_image} 

            camera_id = self.camera_id[idx]
            database['camera_id'] = camera_id
            database['label'] = label

            return database
        
        elif self.mode == 'query':
            input_image, rec_image = self.get_image(self.image_names, idx)
            label = self.get_label(self.image_labels, idx)
            
            if self.query_gray:
                input_image = self.normalize(to_gray(input_image)/255.0)
            else:
                input_image = self.normalize(input_image/255.0)

            database = {'image': input_image} 

            camera_id = self.camera_id[idx]
            database['camera_id'] = camera_id
            database['label'] = label
            
            return database
            
            
        else:
            inds = self.hash_table[idx]
            if len(inds) < self.im_per_id:
                inds = np.random.choice(inds, self.im_per_id, replace=True)
            else:
                inds = np.random.choice(inds, self.im_per_id, replace=False)

            input_image_list = []
            input_rec_list = []
            for id_ in inds:
                input_image, rec_image = self.get_image(self.image_names, id_)
                
                if random.random() < self.train_gray_p:
                    input_image_list.append(self.normalize(to_gray(input_image)/255.0).unsqueeze(0))
                    input_rec_list.append(self.normalize(to_gray(rec_image)/255.0).unsqueeze(0))
                else:
                    input_image_list.append(self.normalize(input_image/255.0).unsqueeze(0))
                    input_rec_list.append(self.normalize(rec_image/255.0).unsqueeze(0))
                

            input_image_list = torch.cat(input_image_list)
            input_rec_list = torch.cat(input_rec_list)

            label = Variable(torch.from_numpy(np.array([idx] * self.im_per_id, dtype='int32')).long())

            database = {'image': input_image_list} 

            if self.mode == 'source':
                database['label'] = label
                database['rec_image'] = input_rec_list #rec_image

            elif self.mode == 'train':
                database['label'] = label
                database['rec_image'] = input_rec_list #rec_image

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
 
        if self.mode == 'train' or self.mode == 'source':
            """ Flip Image """
            if random.random() > 0.5:
                image = np.flip(image, 1) 

            """ Random Cropping """
            if self.random_crop:
                h,w,c = image.shape
                top = np.random.randint(h/8)
                bottom = int(7*h/8+np.random.randint(h/8))
                left = np.random.randint(w/8)
                right = int(7*w/8+np.random.randint(w/8))
                image = image[top:bottom, left:right, :]
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))

        #downsampled_image = self.downsample(image, self.downsample_scale[idx])
        downsampled_image = self.downsample(image, self.downsample_scale)
#         downsampled_image = self.downsample(image)

        downsampled_image_var = Variable(downsampled_image, requires_grad=False)
        downsampled_image = self.affineTnf(downsampled_image_var).data.squeeze(0)

        image_var = Variable(image, requires_grad=False)
        image = self.affineTnf(image_var).data.squeeze(0)
        
        return downsampled_image, image

    def downsample(self, image, downsample_scale=1):
        kernel = (downsample_scale, downsample_scale)
        downsampled_image = F.max_pool2d(image, kernel)
        return downsampled_image
 
    def get_label(self, label_list, idx):
        label = label_list[idx]
        label = Variable(torch.from_numpy(np.array(label, dtype='int32')).long())
        return label
