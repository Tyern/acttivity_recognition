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
curr = Path(__file__).resolve().parent
sys.path.insert(0, curr.parent.as_posix())
from model.builder import BaseModel
from datamodule.datamodule import DataModule, FFTDataModule


class CNNModel(BaseModel):
    def __init__(self, hidden_size=128, sequence_length=256, input_size=42, cnn_filter_size=64, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)

        self.cnn1 = nn.Sequential(
            nn.Conv1d(sequence_length, cnn_filter_size, kernel_size=5, padding="same"),
            nn.BatchNorm1d(num_features=cnn_filter_size),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(cnn_filter_size, hidden_size, kernel_size=5, padding="same"),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        
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
        output = self.cnn1(x)
        output = self.cnn2(output)
        b, _, _ = output.shape
        
        output = self.gap(output).view(b, -1)

        seq_1_output = self.seq_1(output)
        seq_2_output = self.seq_2(output)
        
        output = torch.concat([output, seq_1_output, seq_2_output], dim=1)
        output = self.classifier(output)
        
        return output
    

class FixedCNNModel1(BaseModel): # simplify the structure
    def __init__(self, hidden_size=128, sequence_length=256, input_size=42, cnn_filter_size=64, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)

        self.cnn1 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filter_size, kernel_size=9, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=cnn_filter_size),
            nn.Dropout1d(p=0.2),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(cnn_filter_size, hidden_size, kernel_size=9, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size//2),
            nn.BatchNorm1d(num_features=hidden_size//2),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(in_features=hidden_size//2, out_features=output_size)
        
    def forward(self, x):
        output = self.cnn1(x)
        output = self.cnn2(output)
        b, _, _ = output.shape
        
        output = self.gap(output).view(b, -1)

        output = self.seq_1(output)
        output = self.classifier(output)
        
        return output
    

class FixedCNNModel2(BaseModel): # reduce hidden size
    def __init__(self, hidden_size=64, sequence_length=256, input_size=42, cnn_filter_size=64, output_size=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)

        self.cnn1 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filter_size, kernel_size=9, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=cnn_filter_size),
            nn.Dropout1d(p=0.2),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(cnn_filter_size, hidden_size, kernel_size=9, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.seq_1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size//2),
            nn.BatchNorm1d(num_features=hidden_size//2),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(in_features=hidden_size//2, out_features=output_size)
        
    def forward(self, x):
        output = self.cnn1(x)
        output = self.cnn2(output)
        b, _, _ = output.shape
        
        output = self.gap(output).view(b, -1)

        output = self.seq_1(output)
        output = self.classifier(output)
        
        return output
    

if __name__ == "__main__":
    train_model_class = [CNNModel, FixedCNNModel1, FixedCNNModel2]
    
    n_epochs = 30
    patience = 8

    missing = 2
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
            
            default_root_dir = (curr/"05_lightning_log")
            default_root_dir.mkdir(exist_ok=True)
            trainer = L.Trainer(
                default_root_dir=default_root_dir.as_posix(),
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
    