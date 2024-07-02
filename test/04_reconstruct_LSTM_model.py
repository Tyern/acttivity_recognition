import os 
import json

import torch
from torch import nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import sys
from pathlib import Path
curr = Path(__file__).resolve()
sys.path.insert(0, curr.parent.parent.as_posix())
from model.builder import BaseModel
from datamodule.datamodule import DataModule, FFTDataModule

class FixedLSTMModel1(BaseModel):
    def __init__(self, hidden_size=128, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        
        self.rnn = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=3,
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
        activation, _ = self.rnn(x)
        
        b, _, _ = activation.size()
        lstm_output = activation[:,-1,:].view(b,-1)
        seq_1_output = self.seq_1(lstm_output)
        seq_2_output = self.seq_2(lstm_output)
        
        output = torch.concat([lstm_output, seq_1_output, seq_2_output], dim=1)
        output = self.classifier(output)
        
        return output
    

class FixedLSTMModel2(BaseModel): # test whether simplify the model is better?
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        
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
        # out = self.activation(out)
        out = self.seq_1(out)
        out = self.classifier(out)
        
        return out
    

class FixedLSTMModel3(BaseModel): # test whether add ReLU after rnn is better?
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        
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
        out = self.seq_1(out)
        out = self.classifier(out)
        
        return out
    

class FixedLSTMModel4(BaseModel): # test whether use 2 LSTM is better?
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        
        self.rnn = nn.LSTM(input_size=input_size, 
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
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(in_features=hidden_size, out_features=output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        activation, _ = self.rnn(x)
        b, _, _ = activation.size()
        out = activation[:,-1,:].view(b,-1)
        out = self.activation(out)
        out = self.seq_1(out)
        out = self.classifier(out)
        
        return out

if __name__ == "__main__":
    # train_model_class = [FixedLSTMModel1, FixedLSTMModel2, FixedLSTMModel3, FixedLSTMModel4]
    train_model_class = [FixedLSTMModel1]
    
    n_epochs = 1
    patience = 5

    missing = 0
    user = 0

    batch_size = 256
    for model_class in train_model_class: 
        
        model_name = model_class.__name__
        print("Running for model", model_name)
        for user in [0,1]: ## Use only 2 user
            print("Running for user ", user)
            
            data_module = DataModule(
                test_user=user, 
                missing_sensor_numbers=missing,
                batch_size=batch_size)
            
            net = model_class(input_size=42,
                              output_size=10,
                              sequence_length=256,
                              )
                        
            model_summary = ModelSummary(net, max_depth=6)
            print(model_summary)
                
            print(f"\n*************training on User{user}*************")
            
            trainer = L.Trainer(
                default_root_dir=(curr/"04_lightning_log").as_posix(),
                callbacks=[EarlyStopping(monitor="val_loss", patience=patience)],
                max_epochs=n_epochs,
                check_val_every_n_epoch=1,
                accelerator="gpu", 
                )

            trainer.fit(model=net, datamodule=data_module)
            trainer_test_dict = trainer.logged_metrics

            trainer.test(model=net, datamodule=data_module)
            trainer_test_dict.update(trainer.logged_metrics)
            
            for key in trainer_test_dict.keys():
                trainer_test_dict[key] = trainer_test_dict[key].item()
            
            with open(os.path.join(trainer.logger.log_dir, f"{model_name}_user{user}_missing{missing}.json"), "w") as f:
                json.dump(trainer_test_dict, f, indent=4)
    