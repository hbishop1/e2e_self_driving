from models import *
from utils import *

data_directory = "/home2/pwkw48/4th_year_project/comma_dataset/"
dset = Comma_dataset
lr = 0.001
num_epochs = 100
weight_decay = 0.001
model = Original_PilotNet
stereo = False
weighted_loss = False
batch_size = 16
