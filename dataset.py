# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import os
from skimage import io
from skimage.transform import resize
import numpy as np
import random
import cfg
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CSNet_dataset(Dataset):
    def __init__(self, cfg, torp, transforms = None):
        
        self.data_dir = cfg.data_dir
        self.i_s_dir = cfg.i_s_dir
        self.batch_size = cfg.batch_size
        self.data_shape = cfg.data_shape
        self.torp = torp

        if(self.torp == 'train'):
            self.name_list = os.listdir(os.path.join(self.data_dir, cfg.train_data_dir, self.i_s_dir))
        
        elif(self.torp == 'test'):
            self.name_list = os.listdir(os.path.join(self.data_dir, cfg.test_data_dir, self.i_s_dir))
        
      
    def __len__(self):

        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        if(self.torp == 'train'):
            img_name = self.name_list[idx]
        
            i_s = io.imread(os.path.join(cfg.data_dir, cfg.train_data_dir, cfg.i_s_dir, img_name))
            mask_t = io.imread(os.path.join(cfg.data_dir, cfg.train_data_dir, cfg.mask_t_dir, img_name), as_gray = True)
            
        elif(torp == 'test'):
        
            img_name = self.name_list[idx]
            
            i_s = io.imread(os.path.join(cfg.data_dir, cfg.test_data_dir, cfg.i_s_dir, img_name))
            mask_t = io.imread(os.path.join(cfg.data_dir, cfg.test_data_dir, cfg.mask_t_dir, img_name), as_gray = True)
        
        return [i_t, mask_t]
        

class Example_dataset(Dataset):
    
    def __init__(self, data_dir = cfg.example_data_dir, transform = None):
        
        self.name_list = os.listdir(os.path.join(data_dir, self.i_s_dir))
        # self.files = os.listdir(data_dir)
        # self.files = [i.split('_')[0] + '_' for i in self.files]
        # self.files = list(set(self.files))
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]
        
        i_s = io.imread(os.path.join(cfg.example_data_dir, cfg.i_s_dir, img_name))
        
        h, w = i_s.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        to_h = cfg.data_shape[0]
        to_w = int(round(int(w * scale_ratio) / 8)) * 8
        to_scale = (to_h, to_w)
        
        i_s = resize(i_s, to_scale, preserve_range=True)
        
        sample = (i_s, img_name)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
        
class To_tensor(object):
    def __call__(self, sample):
        
        i_s, img_name = sample

        i_s = i_s.transpose((2, 0, 1)) /127.5 -1

        i_s = torch.from_numpy(i_s)

        return (i_s.float(), img_name)