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
import shutil 
import inspect
from datetime import datetime 
from models import *
from utils import *


if __name__ == '__main__':

    import params

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    os.mkdir(os.path.join(ROOT_DIR,'experiments',time))
    shutil.copy(params.__file__, os.path.join(ROOT_DIR,'experiments',time,'params.py'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataloaders = {x: torch.utils.data.DataLoader(
        params.dset(os.path.join(params.data_directory,x), transform=torchvision.transforms.ToTensor()),
        shuffle=False, batch_size=params.batch_size
        )
    for x in ['train', 'valid']}

    if 'stereo' in inspect.getargspec(params.model)[0]:
        Model = params.model(params.stereo).to(device)
    else:
        Model = params.model().to(device)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    Model.apply(weights_init)

    print(f'Time started: {time}')
    print(f'Number of network parameters: {len(torch.nn.utils.parameters_to_vector(Model.parameters()))}')
    print(f'Training set size: {len(dataloaders["train"].dataset)}')
    print(f'Validation set size: {len(dataloaders["valid"].dataset)}')

    optimiser = torch.optim.Adam(Model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    criterion = Steering_loss()
    logs = {'train_loss':[],'valid_loss':[]}

    for epoch in range(1,params.num_epochs+1):

        print('-' * 10)
        print(f'Epoch {epoch}/{params.num_epochs} \n')
        
        # arrays for metrics
        train_loss_arr = []
        valid_loss_arr = []

        for phase in ['train', 'valid']:
            if phase == 'train':
                criterion.train = True
                Model.train()  # Set model to training mode
            else:
                criterion.train = False
                Model.eval()   # Set model to evaluate mode

            # Iterate over data.
            for i, dat in enumerate(dataloaders[phase]):

                if params.stereo:
                    left, right = dat[0].to(device), dat[1].to(device)
                else:
                    image = dat[0].to(device)

                # zero the parameter gradients
                optimiser.zero_grad()   

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = Model(left, right) if params.stereo else Model(image)
                    loss = criterion.forward(outputs, dat[-1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimiser.step()
                        train_loss_arr.append(loss.item())
                    else:
                        valid_loss_arr.append(loss.item())

                progress = int(i/(len(dataloaders[phase].dataset)/params.batch_size)*50)
                print(f'Epoch {phase} progress: [{"#"*progress}{" "*(50-progress)}]',end='\r')
            
            print(f'\033[K\n{phase} Loss: {np.mean([train_loss_arr,valid_loss_arr][int(phase == "valid")]):.4f} \n')


        logs['train_loss'].append(np.mean(train_loss_arr))
        logs['valid_loss'].append(np.mean(valid_loss_arr))

        with open(os.path.join(ROOT_DIR, 'experiments',time,'train_logs.p'), 'wb') as fp:
            pickle.dump(logs, fp)

        torch.save(Model.state_dict(), os.path.join(ROOT_DIR, 'experiments', time, 'model.pt'))

        with open(os.path.join(ROOT_DIR, 'experiments',time,'results.txt'), 'w') as f:
            f.write(f'Epochs completed: {epoch} \n')
            f.write(f'Best train loss: {min(logs["train_loss"])} \n')
            f.write(f'Best validation loss: {min(logs["valid_loss"])} \n')



