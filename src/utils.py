from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
import cv2
import pandas
import os
import numpy as np
import h5py
import random
import shutil

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

            multiplier = 1 if not (self.train and self.weighted) else (0.5 + abs(target[i,0] / self.max_angle))

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

        return left_image, right_image, steering_commands[:1]


class Comma_dataset(Dataset):
    def __init__(self, root_dir, transform=None, return_restart=False, filtered=True):

        self.transform = transform
        self.root_dir = root_dir
        self.image_files = sorted([os.path.join(root_dir,'camera', i) for i in os.listdir(os.path.join(root_dir,'camera'))])
        self.log_files = sorted([os.path.join(root_dir,'log',i) for i in os.listdir(os.path.join(root_dir,'log')) if filtered == ('-edited' in i)])
        self.im_file_lengths = {}
        self.count = 0
        self.return_restart = return_restart

        for i in self.image_files:
            with h5py.File(i,'r') as f:
                self.im_file_lengths[i] = f['X'].len()
        
        self.log_file_lengths = {}

        total_steering_angle = 0
        total_variance = 0
        total_whiteness = 0
        total_absolute_angle = 0
        total_num = 0

        for i in self.log_files:
            with h5py.File(i,'r') as f:
                arr = np.zeros(f['steering_angle'].shape)
                f['steering_angle'].read_direct(arr)
                self.log_file_lengths[i] = len(arr)
                total_steering_angle += np.sum(arr)
                total_variance += np.sum(np.apply_along_axis(lambda x : x **2,0,arr))
                total_whiteness += np.sum([(arr[j] - arr[j+1])**2 for j in range(len(arr)-2)])
                total_absolute_angle += np.sum(np.apply_along_axis(abs,0,arr))
                total_num += len(arr)

        self.steering_mean = total_steering_angle / total_num
        self.steering_variance = total_variance / total_num - self.steering_mean ** 2
        self.steering_whiteness = (total_whiteness / total_num) ** 0.5
        self.steering_mean_abs_angle = total_absolute_angle / total_num
    

    def __getitem__(self, idx):

        remaining_idx = idx
        file_index = 0
        restart = True

        current_image_count = self.log_file_lengths[self.log_files[file_index]]
        while remaining_idx - current_image_count >= 0:
            remaining_idx -= current_image_count
            file_index += 1
            current_image_count = self.log_file_lengths[self.log_files[file_index]]

        with h5py.File(self.image_files[file_index],'r') as f_im:
            with h5py.File(self.log_files[file_index],'r') as f_log:
                im_ptr = f_log['cam1_ptr'][remaining_idx]
                image = f_im['X'][im_ptr]
                image = np.moveaxis(image, 0, 2)
                steering_angle = f_log['steering_angle'][remaining_idx]
                if remaining_idx != 0:
                    if f_log['cam1_ptr'][remaining_idx - 1] == f_log['cam1_ptr'][remaining_idx] - 1:
                        restart = False

        steering_commands = np.array([steering_angle])

        if self.transform is not None:
            image = self.transform(image)

        if self.return_restart:
            return image, steering_commands, restart
        else:
            return image, steering_commands

    def __len__(self):
        return sum(self.log_file_lengths.values()) 




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
    comma = Comma_dataset("/home2/pwkw48/4th_year_project/comma_dataset/train")
    for a in dir(comma):
        if not a.startswith('__'):
            print(f'{a}:  {getattr(comma, a)}')
