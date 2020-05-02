from models import *
from utils import *

data_directory = "/home2/pwkw48/4th_year_project/stereo_dataset/"
train_dset = Stereo_steering_dataset
test_dset = Stereo_steering_dataset
lr = 3e-4
num_epochs = 200
weight_decay = 0.001
dropout_conv = 0.75
dropout_fc = 0.75
model = Original_PilotNet
stereo = False
weighted_loss = False
batch_size = 16
pre_train_path = None #'/home2/pwkw48/4th_year_project/experiments/best_comma_run/model.pt'
reinitialise_fc = False
kalman_model_path = '/home2/pwkw48/4th_year_project/experiments/01-05-2020_11-44-44/model.pt'