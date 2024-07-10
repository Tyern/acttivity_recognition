
import torch
from torch import nn
import torch.nn.functional as F

import lightning as L
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

from lightning.pytorch.utilities.model_summary import ModelSummary

import os
import sys
curr = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.abspath(curr), "../.."))
from model.builder import BaseModel
from datamodule.datamodule import DataModule, FFTDataModule
    
class CompDiscNet(BaseModel): # test whether add ReLU after rnn is better?
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10):
        self.seq = nn.Sequential(
            nn.Conv2d(1, 64, (5, 5), 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(64, 128, (3, 3), 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
                      
            nn.Conv2d(128, 256, (3, 3), 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            
            nn.Conv2d(256, 256, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            
            nn.Conv2d(256, 256, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(256, 256, (3, 2), 1, (2, 1), dilation=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(256, 256, (3, 2), 1, (4, 2), dilation=(4,4)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            
            nn.Conv2d(256, 256, (3, 2), 1, (8, 4), dilation=(8,8)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            
            nn.Conv2d(256, 256, (3, 2), 1, (16, 8), dilation=(16, 16)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
                                               
            nn.Conv2d(256, 256, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(256, 256, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            nn.ConvTranspose2d(256, 128, (4, 4), 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(128, 128, (3, 4), 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.ConvTranspose2d(128, 64, (4, 4), 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(64, 32, (3, 3), 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(32, 16, (3, 4), 1, (1, 1)),
            nn.Sigmoid(),
        )
        
