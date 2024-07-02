from functools import reduce
import numpy as np
import torch
from torch import nn

import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary

import sys
from pathlib import Path
curr = Path(__file__).resolve()
sys.path.insert(0, curr.parent.parent.as_posix())
from model.builder import ConvLSTMCell, ConvLSTM, BaseModel


class ConvLSTMModel(BaseModel):
    def __init__(
        self, 
        hidden_size=64, 
        sequence_length=256, 
        cnn_filter_size=64, 
        input_size=42,
        output_size=10, 
        **kwargs):

        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        
        self.conv1 = nn.Conv2d(1, cnn_filter_size//2, (3, 3), 1, "same")
        self.bn1 = nn.BatchNorm2d(cnn_filter_size//2)
        self.conv2 = nn.Conv2d(cnn_filter_size//2, cnn_filter_size, (3, 3), 1, "same")
        self.bn2 = nn.BatchNorm2d(cnn_filter_size)
        self.mp = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_lstm = ConvLSTM(input_dim=(cnn_filter_size), hidden_dim=hidden_size, kernel_size=(1,3), num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64*128*21, 64)
        self.fc2 = nn.Linear(64, output_size)
        
        # Activation
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        
        out = self.conv1(x)
        out = self.activation(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.bn2(out)
        out = self.mp(out)
        
        out = out.view(x.shape[0], 1, 64, 128, 21)
        _, last_state_list = self.conv_lstm(out)
        out = last_state_list[-1][0] # last state ht
        out = self.activation(out)
        
        out = out.view(x.shape[0], -1)
        out = self.fc1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    
    
class ConvLSTMModelAttention(BaseModel):
    def __init__(
        self, 
        hidden_size=64, 
        sequence_length=256, 
        cnn_filter_size=64, 
        input_size=42,
        output_size=10, 
        **kwargs):

        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        
        self.conv1 = nn.Conv2d(1, cnn_filter_size//2, (3, 3), 1, "same")
        self.bn1 = nn.BatchNorm2d(cnn_filter_size//2)
        self.conv2 = nn.Conv2d(cnn_filter_size//2, cnn_filter_size, (3, 3), 1, "same")
        self.bn2 = nn.BatchNorm2d(cnn_filter_size)
        self.mp = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_lstm = ConvLSTM(input_dim=(cnn_filter_size), hidden_dim=hidden_size, kernel_size=(1,3), num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64*128*21, 64)
        self.fc2 = nn.Linear(64, output_size)
        
        # Activation
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        
        out = self.conv1(x)
        out = self.activation(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.bn2(out)
        out = self.mp(out)
        
        out = out.view(x.shape[0], 1, 64, 128, 21)
        _, last_state_list = self.conv_lstm(out)
        out = last_state_list[-1][0] # last state ht
        out = self.activation(out)
        
        out = out.view(x.shape[0], -1)
        out = self.fc1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        out = self.softmax(out)
        return out