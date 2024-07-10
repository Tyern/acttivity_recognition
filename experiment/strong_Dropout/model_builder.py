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
    

class FixedLSTMModel3(BaseModel): # test whether add ReLU after rnn is better?
    def __init__(self, 
                 hidden_size=64, 
                 sequence_length=256, 
                 input_size=42, 
                 output_size=10, 
                 dropout_rate=0.2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        self.rnn = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
        )
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(in_features=hidden_size, out_features=output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        activation, _ = self.rnn(x)
        b, _, _ = activation.size()
        out = activation[:,-1,:].view(b,-1)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.seq_1(out)
        out = self.classifier(out)
        
        return out
    
    
if __name__ == "__main__":
    model_class_list = [FixedLSTMModel3]
    for model_class in model_class_list:
        model_name = model_class.__name__
        print("#"*20, model_name, "#"*20)
        model = model_class()
        data_module = DataModule(
                test_user=0, 
                missing_sensor_numbers=0,
                batch_size=10)
        
        print(ModelSummary(model))
        trainer = L.Trainer(
            accelerator="gpu", 
            max_steps=5,
            enable_checkpointing=False, 
            logger=False)
        
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        
        print("trainer.logged_metrics", trainer.logged_metrics)
        
        
        
        
        
        