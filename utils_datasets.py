#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:45:35 2022

@author: chingis
"""
from glob import glob
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from os.path import exists
def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img
class PetDataset(Dataset):
    "Dataset to serve individual images to our model"
    
    def __init__(self, data, transforms=None):
        self.data = data
        self.len = len(data)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img, label = self.data[index]
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
    
    def __len__(self):
        return self.len


# Since the data is not split into train and validation datasets we have to 
# make sure that when splitting between train and val that all classes are represented in both
class Databasket(object):
    "Helper class to ensure equal distribution of classes in both train and validation datasets"
    
    def __new__(cls, data_dir, val_split=0.2, download=False, train=False, transform=None):
        if download:
            if not train:
                return
            #if exists(f'{data_dir}/train.pt') and exists(f'{data_dir}/val.pt'):
             #   print('exists')
              #  return
            filenames = glob(f'{data_dir}/*.jpg')
            classes = set()
            
            data = []
            labels = []
            
            # Load the images and get the classnames from the image path
            for image in filenames:
                class_name = image.rsplit("/", 1)[1].rsplit('_', 1)[0]
                classes.add(class_name)
                img = load_image(image)
            
                data.append(img)
                labels.append(class_name)
            
            # convert classnames to indices
            class2idx = {cl: idx for idx, cl in enumerate(classes)}        
            labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()
            
            data = list(zip(data, labels))
            class_values = [[] for x in range(len(classes))]
            
            # create arrays for each class type
            for d in data:
                class_values[d[1].item()].append(d)
                
            train_data = []
            val_data = []
            print(len(class2idx))
            # put (1-val_split) of the images of each class into the train dataset
            # and val_split of the images into the validation dataset
            for class_dp in class_values:
                split_idx = int(len(class_dp)*(1-val_split))
                train_data += class_dp[:split_idx]
                val_data += class_dp[split_idx:]
            torch.save(train_data, f'{data_dir}/train.pt')
            torch.save(val_data, f'{data_dir}/val.pt')

        else:
            if train:
                train_ds = PetDataset(torch.load(f'{data_dir}/train.pt'),transform)
                return train_ds
            else:
                val_ds = PetDataset(torch.load(f'{data_dir}/val.pt'),transform)
                return val_ds