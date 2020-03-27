from models import *
from utils import *

data_directory = "/home2/pwkw48/4th_year_project/comma_dataset/"
train_dset = Augmented_comma_dataset
test_dset = Comma_dataset
lr = 1e-4
num_epochs = 100
weight_decay = 0.0001
model = Original_PilotNet
stereo = False
weighted_loss = True
batch_size = 32
