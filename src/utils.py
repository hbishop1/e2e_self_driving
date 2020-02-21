from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import pandas
import os
import math
import ast
import numpy as np
import pickle
import argparse
import sys
import h5py
from datetime import datetime 
from utils import *

class Steering_loss():

    def __init__(self, train = False, weighted = False):
        super(Steering_loss,self).__init__()
        self.train = train
        self.weighted = weighted

    def forward(self,outputs,target):

        total_loss = 0
        batch_size = outputs.size()[0]

        for i in range(batch_size):

                multiplier = 1 if not (self.train and self.weighted) else (0.2 + abs(target[i,0]))

                total_loss += sum([(outputs[i,j] - target[i,j]) ** 2 for j in range(2)]) * multiplier

        batch_loss = total_loss / batch_size

        return batch_loss


class Stereo_steering_dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.steering_df = pandas.read_csv(os.path.join(self.root_dir,'steering.csv'))
        self.transform = transform

    def __len__(self):
        return self.steering_df.shape[0]

    def __getitem__(self, idx):

        root_filename = self.steering_df.iloc[idx,1]
        steering_commands = np.array([self.steering_df.iloc[idx,2],self.steering_df.iloc[idx,3]])
        left_image = cv2.imread(os.path.join(self.root_dir,'left_camera',root_filename))
        right_image = cv2.imread(os.path.join(self.root_dir,'right_camera',root_filename))

        if self.transform is not None:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, steering_commands


class Comma_dataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.transform = transform
        self.root_dir = root_dir
        self.image_files = sorted([os.path.join(root_dir,'camera', i) for i in os.listdir(os.path.join(root_dir,'camera')) if '11-24-51-edited' in i])
        self.log_files = sorted([os.path.join(root_dir,'log',i) for i in os.listdir(os.path.join(root_dir,'log')) if '11-24-51-edited' in i])
        self.im_file_lengths = {}
        self.count = 0

        for i in self.image_files:
            with h5py.File(i,'r') as f:
                self.im_file_lengths[i] = f['image'].len()


    def __getitem__(self, idx):

        remaining_idx = idx
        file_index = 0

        current_image_count = self.im_file_lengths[self.image_files[file_index]]
        while remaining_idx - current_image_count >= 0:
            remaining_idx -= current_image_count
            file_index += 1
            current_image_count = self.im_file_lengths[self.image_files[file_index]]

        with h5py.File(self.image_files[file_index],'r') as f_im:
            with h5py.File(self.log_files[file_index],'r') as f_log:
                image = f_im['image'][remaining_idx]
                image = cv2.resize(np.moveaxis(image, 0, 2),(672,188))
                steering_angle = f_log['steering_angle'][remaining_idx]
                speed = f_log['speed_abs'][remaining_idx]
    
        steering_commands = np.array([steering_angle,speed])

        print(self.count%100)
        if self.count%100 == 0:
            cv2.imwrite(f"test/{self.count}.png",image)
            with open(f"test/{self.count}.txt",'w') as f:
                f.write(str(steering_commands))
        self.count +=1

        if self.transform is not None:
            image = self.transform(image)

        return image, steering_commands

    def __len__(self):
        return sum(self.im_file_lengths.values())
 


if __name__ == '__main__':
    comma = Comma_dataset("/home2/pwkw48/4th_year_project/comma_dataset")
    mine = Stereo_steering_dataset("/home2/pwkw48/4th_year_project/data/train/")
    print(comma.__getitem__(1000)[0].shape)
    cv2.imwrite("test.png",comma.__getitem__(1000)[0])