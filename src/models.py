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
import h5py
from datetime import datetime 
from utils import *


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
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

        self.regression_layers.append(nn.Linear(in_features=256, out_features=2))

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


