#!/usr/bin/env python3

try:
    from torch.utils.data import Dataset
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

except ImportError:
    print('Couldnt import pytorch')

import os
import rosbag
import cv2
import pandas
import math
import ast
import numpy as np
import pickle
import argparse

class Steering_loss():

    def __init__(self):
        super(Steering_loss,self).__init__()

    def forward(self,outputs,target):

        total_loss = 0
        batch_size = outputs.size()[0]

        for i in range(batch_size):

                total_loss += sum([(outputs[i,j] - target[i,j]) ** 2 for j in range(2)])

        batch_loss = total_loss / batch_size

        return batch_loss


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class My_PilotNet(nn.Module):
    def __init__(self):

        super(My_PilotNet, self).__init__()
        self.feature_extract_layers = nn.ModuleList()
        self.regression_layers = nn.ModuleList()

        # input image 3 x 188 x 672
        self.feature_extract_layers.append(nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=2))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(24))
        self.feature_extract_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # 24 x 94 x 336
        self.feature_extract_layers.append(nn.Conv2d(24, 36, kernel_size=5, stride=1, padding=2))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(36))
        self.feature_extract_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # 36 x 47 x 168
        self.feature_extract_layers.append(nn.Conv2d(36, 48, kernel_size=5, stride=1, padding=2))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(48))
        self.feature_extract_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # 48 x 23 x 84
        self.feature_extract_layers.append(nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(64))
        self.feature_extract_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # 64 x 11 x 42
        self.feature_extract_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(64))
        self.feature_extract_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.regression_layers.append(Flatten())

        self.regression_layers.append(nn.Linear(in_features=64*5*21, out_features=1024))
        self.regression_layers.append(nn.ReLU())
        self.regression_layers.append(nn.BatchNorm1d(1024))

        self.regression_layers.append(nn.Linear(in_features=1024, out_features=256))
        self.regression_layers.append(nn.ReLU())
        self.regression_layers.append(nn.BatchNorm1d(256))

        self.regression_layers.append(nn.Linear(in_features=256, out_features=2))

    def forward(self, left, right):
        x = left
        for m in self.feature_extract_layers:
            x = m(x)
        for m in self.regression_layers:
            x = m(x)
        return x


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, help='location of the data')
    parser.add_argument('--lr', type=float, help='learning rate', default = 0.001)
    parser.add_argument('--num_epochs', type=int, help='number of training epochs', default=100)
    parser.add_argument('--weight_decay', type=float, help='weight decay factor', default=0.005)

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataloaders = {x: torch.utils.data.DataLoader(
        Stereo_steering_dataset(os.path.join(args.data_directory,x), transform=torchvision.transforms.ToTensor()),
        shuffle=False, batch_size=8
        )
    for x in ['train', 'valid']}

    Model = My_PilotNet().to(device)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    Model.apply(weights_init)

    print('Number of network parameters: ', len(torch.nn.utils.parameters_to_vector(Model.parameters())))

    optimiser = torch.optim.Adam(Model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Steering_loss()
    logs = {'train_loss':[],'valid_loss':[]}

    for epoch in range(1,args.num_epochs+1):

        print('-' * 10)
        print('Epoch {}/{}'.format(epoch,args.num_epochs))
        
        # arrays for metrics
        train_loss_arr = np.zeros(0)
        valid_loss_arr = np.zeros(0)


        for phase in ['train', 'test']:
            if phase == 'train':
                Model.train()  # Set model to training mode
            else:
                Model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for left, right, t in dataloaders[phase]:
                left, right = left.to(device), right.to(device)

                # zero the parameter gradients
                optimiser.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = Model(left, right)
                    loss = criterion.forward(outputs, t)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimiser.step()

        print('Train Loss: {:.4f} \n'.format(train_loss_arr.mean()))
        print('Validation Loss: {:.4f} \n'.format(valid_loss_arr.mean()))

        logs['train_loss'].append(train_loss_arr.mean())
        logs['valid_loss'].append(valid_loss_arr.mean())

        with open('logs.p', 'wb') as fp:
            pickle.dump(logs, fp)







    
