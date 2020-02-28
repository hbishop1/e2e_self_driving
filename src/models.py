import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class Original_PilotNet(nn.Module):
    def __init__(self,stereo=True):

        super(Original_PilotNet, self).__init__()

        self.stereo = stereo
        self.feature_extract_layers = nn.ModuleList()
        self.regression_layers = nn.ModuleList()

        # input image 3 x 66 x 200
        self.feature_extract_layers.append(nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(24))

        # 24 x 31 x 98
        self.feature_extract_layers.append(nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(36))

        # 36 x 14 x 47
        self.feature_extract_layers.append(nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(48))

        # 48 x 5 x 22
        self.feature_extract_layers.append(nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(64))

        # 64 x 3 x 20
        self.feature_extract_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))
        self.feature_extract_layers.append(nn.ReLU())
        self.feature_extract_layers.append(nn.BatchNorm2d(64))

        # 64 x 1 x 18
        self.feature_extract_layers.append(Flatten())

        self.regression_layers.append(nn.Linear(in_features=64*1*18 * (int(stereo)+1), out_features=100))
        self.regression_layers.append(nn.ReLU())

        self.regression_layers.append(nn.Linear(in_features=100, out_features=50))
        self.regression_layers.append(nn.ReLU())

        self.regression_layers.append(nn.Linear(in_features=50, out_features=1))

    def forward(self, left, right=None):
        x = F.interpolate(left, (66,200))
        for m in self.feature_extract_layers:
            x = m(x)
        if self.stereo:
            y = F.interpolate(right, (66,200))
            for m in self.feature_extract_layers:
                y = m(y)
            x = torch.cat((x,y), 1)
        for m in self.regression_layers:
            x = m(x)
        return x


class My_PilotNet(nn.Module):
    def __init__(self,stereo=True):

        super(My_PilotNet, self).__init__()

        self.stereo = stereo
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

        self.feature_extract_layers.append(Flatten())

        self.regression_layers.append(nn.Linear(in_features=64*5*21 * (int(stereo)+1), out_features=1024))
        self.regression_layers.append(nn.ReLU())
        self.regression_layers.append(nn.BatchNorm1d(1024))

        self.regression_layers.append(nn.Linear(in_features=1024, out_features=256))
        self.regression_layers.append(nn.ReLU())
        self.regression_layers.append(nn.BatchNorm1d(256))

        self.regression_layers.append(nn.Linear(in_features=256, out_features=1))

    def forward(self, left, right=None):
        x = left
        for m in self.feature_extract_layers:
            x = m(x)
        if self.stereo:
            y = right
            for m in self.feature_extract_layers:
                y = m(y)
            x = torch.cat((x,y), 1)
        for m in self.regression_layers:
            x = m(x)
        return x
