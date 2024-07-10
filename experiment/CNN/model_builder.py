
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
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        self.dropout = nn.Dropout2d(p=0.2)
        
        self.rnn = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
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
    
    
class FixedLSTMModel5(BaseModel): # test whether add ReLU after rnn is better?
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        self.dropout = nn.Dropout2d(p=0.2)
        
        self.rnn = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=3,
                          batch_first=True)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
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


class FixedLSTMModel3Attention1(BaseModel): # test whether add ReLU after rnn is better?
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
        
        self.rnn1 = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.attention1 = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(input_size=hidden_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
        )
        self.classifier = nn.Linear(in_features=hidden_size, out_features=output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        output, _ = self.rnn1(x)
        output, _ = self.attention1(output, output, output)
        
        output = self.activation(output)
        output = self.dropout(output)
        
        output, _ = self.rnn2(output)
        b, _, _ = output.size()
        
        output = self.activation(output)
        output = self.dropout(output)
        
        output = output[:,-1,:].view(b,-1)
        
        output = self.seq_1(output)
        output = self.classifier(output)
        
        return output
    

class FixedLSTMModel5Attention1(BaseModel): # test whether add ReLU after rnn is better?
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
        
        self.rnn1 = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.attention1 = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=hidden_size//2,
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(input_size=hidden_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.attention2 = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=hidden_size//2,
            batch_first=True
        )
        
        self.rnn3 = nn.LSTM(input_size=hidden_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.attention3 = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=hidden_size//2,
            batch_first=True
        )
        
        self.rnn4 = nn.LSTM(input_size=hidden_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
        )
        self.classifier = nn.Linear(in_features=hidden_size, out_features=output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        output, _ = self.rnn1(x)
        output, _ = self.attention1(output, output, output)
        
        output = self.activation(output)
        output = self.dropout(output)
        
        output, _ = self.rnn2(output)
        output, _ = self.attention2(output, output, output)
        
        output = self.activation(output)
        output = self.dropout(output)
        
        output, _ = self.rnn3(output)
        output, _ = self.attention3(output, output, output)
        
        output = self.activation(output)
        output = self.dropout(output)
        
        output, _ = self.rnn4(output)
        b, _, _ = output.size()
        
        output = self.activation(output)
        output = self.dropout(output)
        
        output = output[:,-1,:].view(b,-1)
        
        output = self.seq_1(output)
        output = self.classifier(output)
        
        return output
    

class FixedLSTMModel3_GAttention1(BaseModel): # test whether add ReLU after rnn is better?
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
        
        # input should be unsqueeze
        self.cnn_extract1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,3), stride=(1,3))
        # output should be squeeze channel, to (b, 256, 42//3)
        self.rnn1 = nn.LSTM(input_size=input_size, 
                          hidden_size=14,
                          num_layers=1,
                          batch_first=True)
        
        self.rnn2 = nn.LSTM(input_size=14, 
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
        )
        self.classifier = nn.Linear(in_features=hidden_size, out_features=output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # cnn path
        cnn_extract = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        cnn_extract = self.cnn_extract1(cnn_extract)
        cnn_extract = cnn_extract.view(cnn_extract.shape[0], cnn_extract.shape[2], cnn_extract.shape[3])
        
        # rnn path
        output, _ = self.rnn1(x)
        
        # attention and residual
        attn = nn.functional.scaled_dot_product_attention(output, cnn_extract, cnn_extract)
        output = attn + output
        
        output, _ = self.rnn2(output)
        
        output = self.activation(output)
        output = self.dropout(output)
        
        b, _, _ = output.size()
        output = output[:,-1,:].view(b,-1)
        
        output = self.seq_1(output)
        output = self.classifier(output)
        
        return output
    

class FixedLSTMModel3_GAttention2(BaseModel): # test whether add ReLU after rnn is better?
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
        
        # input should be unsqueeze
        self.cnn_extract1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,6), stride=(1,6))
        # output should be squeeze channel, to (b, 256, 42//6)
        self.rnn1 = nn.LSTM(input_size=input_size, 
                          hidden_size=7,
                          num_layers=1,
                          batch_first=True)
        
        self.rnn2 = nn.LSTM(input_size=7, 
                    hidden_size=hidden_size,
                    num_layers=2,
                    batch_first=True)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
        )
        self.classifier = nn.Linear(in_features=hidden_size, out_features=output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # cnn path
        cnn_extract = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        cnn_extract = self.cnn_extract1(cnn_extract)
        cnn_extract = cnn_extract.view(cnn_extract.shape[0], cnn_extract.shape[2], cnn_extract.shape[3])
        
        # rnn path
        output, _ = self.rnn1(x)
        
        # attention and residual
        attn = nn.functional.scaled_dot_product_attention(output, cnn_extract, cnn_extract)
        output = attn + output
        
        output, _ = self.rnn2(output)
        
        output = self.activation(output)
        output = self.dropout(output)
        
        b, _, _ = output.size()
        output = output[:,-1,:].view(b,-1)
        
        output = self.seq_1(output)
        output = self.classifier(output)
        
        return output
    

class FixedLSTMModel3_GAttention1_Mulpath(BaseModel):
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
        
        # input should be unsqueeze
        self.cnn_extract1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,3), stride=(1,3))
        # output should be squeeze channel, to (b, 256, 42//3)
        self.rnn1 = nn.LSTM(input_size=input_size, 
                          hidden_size=14,
                          num_layers=1,
                          batch_first=True)
        
        self.rnn2 = nn.LSTM(input_size=14, 
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
        )
        
        self.seq_2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(in_features=3 * hidden_size, out_features=output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        # cnn path
        cnn_extract = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        cnn_extract = self.cnn_extract1(cnn_extract)
        cnn_extract = cnn_extract.view(cnn_extract.shape[0], cnn_extract.shape[2], cnn_extract.shape[3])
        
        # rnn path
        output, _ = self.rnn1(x)
        
        # attention and residual
        attn = nn.functional.scaled_dot_product_attention(output, cnn_extract, cnn_extract)
        output = output + attn
        
        output, _ = self.rnn2(output)
        
        output = self.activation(output)
        output = self.dropout(output)
        
        b, _, _ = output.size()
        output = output[:,-1,:].view(b,-1)
        
        seq_1_output = self.seq_1(output)
        seq_2_output = self.seq_2(output)
        
        output = torch.concat([output, seq_1_output, seq_2_output], dim=1)
        output = self.classifier(output)
        
        return output
    
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    model_class_list = [FixedLSTMModel3, FixedLSTMModel3Attention1, FixedLSTMModel3_GAttention1, FixedLSTMModel3_GAttention1_Mulpath, FixedLSTMModel5, FixedLSTMModel5Attention1]
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
            fast_dev_run=True)
        trainer.fit(model, data_module)
        