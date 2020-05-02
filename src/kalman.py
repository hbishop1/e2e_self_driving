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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    
    test_dataset = params.test_dset(os.path.join(params.data_directory,'valid'), transform=transforms, return_restart=True)
    train_dataset = params.train_dset(os.path.join(params.data_directory,'train'), transform=transforms)
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle = False, 
        batch_size=1)

    if 'stereo' in inspect.getargspec(params.model)[0]:
        Model = params.model(stereo=params.stereo,dropout_conv=params.dropout_conv,dropout_fc=params.dropout_fc).to(device)
    else:
        Model = params.model(dropout_conv=params.dropout_conv,dropout_fc=params.dropout_fc).to(device)

    Model.load_state_dict(torch.load(params.kalman_model_path))

    print(f'Time started: {time}')  
    print(f'Number of network parameters: {len(torch.nn.utils.parameters_to_vector(Model.parameters()))}')

    logs = {'mae_kalman':[],'rmse_kalman':[],'whiteness_kalman':[],'mae_no_kalman':[],'rmse_no_kalman':[],'whiteness_no_kalman':[],'kalman_prediction':[],'no_kalman_prediction':[],'gt':[],'sigma':[]}

    Model.eval()   # Set model to evaluate mode

    previous_output = 0
    previous_mut = train_dataset.steering_mean
    m = train_dataset.steering_mean
    previous_sigmat2 = train_dataset.steering_variance
    lambda2 = 230#Model.mse
    alpha2 = train_dataset.steering_whiteness**2
    eta2 = 7000#train_dataset.steering_variance

    print('m: ',m)
    print('lambda2: ',lambda2)
    print('alpha2: ',alpha2)
    print('eta2: ',eta2)



    # Iterate over data.
    for i, dat in enumerate(dataloader):                   

        if params.stereo:
            left, right = dat[0].to(device), dat[1].to(device)
        else:
            image = dat[0].to(device)

        target = dat[-2].float().to(device)

        logs['gt'].append(target.item())

        with torch.set_grad_enabled(False):
            output = Model(left, right) if params.stereo else Model(image)

        logs['no_kalman_prediction'].append(output.item())
        logs['mae_no_kalman'].append(abs(output.item() - target.item()))
        logs['rmse_no_kalman'].append((output.item() - target.item()) ** 2)
        logs["whiteness_no_kalman"].append((output.item() - previous_output) ** 2)

        
        if dat[-1]:
            mut = output.item()
            sigmat2 = lambda2

        else:
            sigmat2 = 1/ (1/lambda2 - 1/eta2 + 1/(alpha2 + previous_sigmat2))
            mut = ((output.item() / lambda2) - (m / eta2) + previous_mut / (alpha2 + previous_sigmat2)) * sigmat2

        
        logs['kalman_prediction'].append(mut)
        logs['mae_kalman'].append(abs(mut - target.item()))
        logs['rmse_kalman'].append((mut - target.item()) ** 2)
        logs["whiteness_kalman"].append((mut - previous_mut ) ** 2)    

        logs['sigma'].append(previous_sigmat2)
        
        previous_output = output.item()
        previous_mut = mut
        previous_sigmat2 = sigmat2

    
    os.mkdir(os.path.join(ROOT_DIR,'experiments',time))
    plt.plot(logs['sigma'][:100])
    plt.savefig(os.path.join(ROOT_DIR, 'experiments', time, 'sigma_fig'))
    plt.clf()
    plt.plot(logs['no_kalman_prediction'][:2000],color='blue')
    plt.plot(logs['gt'][:2000],color='green')
    plt.plot(logs['kalman_prediction'][:2000],color='red')
    plt.savefig(os.path.join(ROOT_DIR, 'experiments', time, 'steering_fig'))
    plt.clf()
    plt.hist(logs['gt'],bins=500)
    plt.savefig(os.path.join(ROOT_DIR, 'experiments', time, 'gt_hist'))
    plt.clf()
    plt.hist(logs['mae_no_kalman'],bins=100)
    plt.savefig(os.path.join(ROOT_DIR, 'experiments', time, 'mae_no_kalman_hist'))
    plt.clf()
    plt.hist([logs['gt'][i] - logs['gt'][i+1] for i in range(len(logs['gt'])-2)],bins=250,range=(-10,10))
    plt.savefig(os.path.join(ROOT_DIR, 'experiments', time, 'change_per_frame_hist'))



    print(f'Kalman MAE: {np.mean(logs["mae_kalman"])}')
    print(f'Kalman RMSE: {np.mean(logs["rmse_kalman"]) ** 0.5}')
    print(f'Kalman Whiteness: {np.mean(logs["whiteness_kalman"]) ** 0.5}')
    print(f'No Kalman MAE: {np.mean(logs["mae_no_kalman"])}')
    print(f'No Kalman RMSE: {np.mean(logs["rmse_no_kalman"]) ** 0.5}')
    print(f'No Kalman Whiteness: {np.mean(logs["whiteness_no_kalman"]) ** 0.5}')

