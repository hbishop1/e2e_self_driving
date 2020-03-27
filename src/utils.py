from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
import cv2
import pandas
import os
import numpy as np
import h5py
import random

def MAE(outputs,targets):
    total = 0
    for i in range(outputs.size()[0]):
        total += abs(outputs[i] - targets[i])
    
    return total.item() / outputs.size()[0]

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Steering_loss():

    def __init__(self, weighted = False):
        super(Steering_loss,self).__init__()
        self.train = True
        self.weighted = weighted
        if weighted:
            from params import train_dset
            if train_dset in [Comma_dataset,Augmented_comma_dataset]:
                self.max_angle = -360
            else:
                self.max_angle = 1

    def forward(self,outputs,target):

        total_loss = 0
        batch_size = outputs.size()[0]

        for i in range(batch_size):

                multiplier = 1 if not (self.train and self.weighted) else (abs(target[i,0] / self.max_angle))

                total_loss += sum([(outputs[i,j] - target[i,j]) ** 2 for j in range(len(outputs[i]))]) * multiplier

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

        # steering commands are of the form [steering_angle, velocity]
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
        self.image_files = sorted([os.path.join(root_dir,'camera', i) for i in os.listdir(os.path.join(root_dir,'camera'))])
        self.log_files = sorted([os.path.join(root_dir,'log',i) for i in os.listdir(os.path.join(root_dir,'log')) if '-edited' in i])
        self.im_file_lengths = {}
        self.count = 0

        for i in self.image_files:
            with h5py.File(i,'r') as f:
                self.im_file_lengths[i] = f['X'].len()
        
        log_file_lengths = {}
        for i in self.log_files:
            with h5py.File(i,'r') as f:
                log_file_lengths[i] = f['steering_angle'].len()

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
                image = f_im['X'][remaining_idx]
                image = np.moveaxis(image, 0, 2)
                steering_angle = min(360.0,max(-360.0,f_log['steering_angle'][remaining_idx]))
                #speed = f_log['speed_abs'][remaining_idx]

        steering_commands = np.array([steering_angle])

        if self.transform is not None:
            image = self.transform(image)

        return image, steering_commands

    def __len__(self):
        return sum(self.im_file_lengths.values()) 


class Augmented_comma_dataset(Comma_dataset):

    def __init__(self, root_dir, transform=None):

        Comma_dataset.__init__(self, root_dir, None) 
        self.child_transform = transform

    def __getitem__(self, i):
        image, target = super(Augmented_comma_dataset, self).__getitem__(i)
        hflip = random.random() < 0.5
        if hflip:
            image = np.flip(image,1)
            image = image.copy()
            target *= -1

        if self.child_transform is not None:
            image = self.child_transform(image)

        return image, target
        



if __name__ == '__main__':
    comma = Comma_dataset("/home2/pwkw48/4th_year_project/comma_dataset")
    mine = Stereo_steering_dataset("/home2/pwkw48/4th_year_project/data/train/")
    print(comma.__getitem__(1000)[0].shape)
    cv2.imwrite("test.png",comma.__getitem__(1000)[0])