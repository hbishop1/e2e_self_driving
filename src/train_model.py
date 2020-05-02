import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
import pickle
import shutil
import inspect
import matplotlib.pyplot as plt
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

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    
    dataloaders = {x: torch.utils.data.DataLoader(
        [params.train_dset,params.test_dset][i](os.path.join(params.data_directory,x), transform=transforms),
        shuffle = False, 
        batch_size=params.batch_size
        )
    for i,x in enumerate(['train', 'valid'])}

    if 'stereo' in inspect.getargspec(params.model)[0]:
        Model = params.model(
            stereo=params.stereo,
            dropout_conv=params.dropout_conv,
            dropout_fc=params.dropout_fc,
            ).to(device)
    else:
        Model = params.model(
            dropout_conv=params.dropout_conv,
            dropout_fc=params.dropout_fc,
            ).to(device)

    if not params.pre_train_path is None:
        if params.reinitialise_fc:
            for layer in Model.regression_layers:
                if isinstance(layer,nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)

    print(f'Time started: {time}')  
    print(f'Number of network parameters: {len(torch.nn.utils.parameters_to_vector(Model.parameters()))}')
    print(f'Training set size: {len(dataloaders["train"].dataset)}')
    print(f'Validation set size: {len(dataloaders["valid"].dataset)}')

    optimiser = torch.optim.AdamW(Model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    criterion = Steering_loss(weighted = params.weighted_loss)
    logs = {'train_loss':[],'valid_loss':[],'train_mae':[],'valid_mae':[]}
    best_valid_loss = np.Inf

    for epoch in range(1,params.num_epochs+1):
        print('-' * 10)
        print(f'Epoch {epoch}/{params.num_epochs} \n')
        
        # arrays for metrics
        train_loss_arr = []
        valid_loss_arr = []
        train_mae_arr = []
        valid_mae_arr = []
        whiteness_arr = []

        for phase in ['train', 'valid']:
            if phase == 'train':
                criterion.train = True
                Model.train()  # Set model to training mode
            else:
                criterion.train = False
                Model.eval()   # Set model to evaluate mode

            # Iterate over data.
            for i, dat in enumerate(dataloaders[phase]):                    
            #for i in range([2500,500][phase == 'valid']):
             #   dat = next(iter(dataloaders[phase]))
            
                if params.stereo:
                    left, right = dat[0].to(device), dat[1].to(device)
                else:
                    image = dat[0].to(device)

                targets = dat[-1].float().to(device)

                # zero the parameter gradients
                optimiser.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = Model(left, right) if params.stereo else Model(image)
                    loss = criterion.forward(outputs, targets)
                    mae = MAE(outputs,targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimiser.step()
                        train_loss_arr.append(loss.item())
                        train_mae_arr.append(mae)
                    else:
                        valid_loss_arr.append(loss.item())
                        valid_mae_arr.append(mae)

            print(f'{phase} Loss: {np.mean([train_loss_arr,valid_loss_arr][int(phase == "valid")]):.4f}')
            print(f'{phase} MAE: {np.mean([train_mae_arr,valid_mae_arr][int(phase == "valid")]):.4f}')


        logs['train_loss'].append(np.mean(train_loss_arr))
        logs['valid_loss'].append(np.mean(valid_loss_arr))
        logs['train_mae'].append(np.mean(train_mae_arr))
        logs['valid_mae'].append(np.mean(valid_mae_arr))

        with open(os.path.join(ROOT_DIR, 'experiments',time,'train_logs.p'), 'wb') as fp:
            pickle.dump(logs, fp)

        if np.mean(valid_loss_arr) < best_valid_loss:
            #Model.mse = torch.tensor([np.mean(train_loss_arr)])
            torch.save(Model.state_dict(), os.path.join(ROOT_DIR, 'experiments', time, 'model.pt'))
            best_valid_loss = np.mean(valid_loss_arr)

        with open(os.path.join(ROOT_DIR, 'experiments',time,'results.txt'), 'w') as f:
            f.write(f'Epochs completed: {epoch} \n')
            f.write(f'Best train loss: {min(logs["train_loss"])} \n')
            f.write(f'Best train MAE: {min(logs["train_mae"])} \n')
            f.write(f'Best validation loss: {min(logs["valid_loss"])} \n')
            f.write(f'Best validation MAE: {min(logs["valid_mae"])} \n')



