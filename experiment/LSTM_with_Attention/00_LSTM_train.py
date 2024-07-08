import lightning as L
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary

import torch.optim as optim

from tqdm.auto import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy  as np
import tracemalloc 

import json
import os
import gc
import time
import sys

from model_builder import FixedLSTMModel3, FixedLSTMModel3Attention1, FixedLSTMModel3_GAttention1, FixedLSTMModel3_GAttention1_Mulpath

sys.path.insert(0, os.path.join(os.path.abspath(''), "../.."))
from model.builder import LSTMModel, LSTMAttentionModel
from datamodule.datamodule import DataModule, FFTDataModule

from pc_control.pc_control import get_PC

import warnings
warnings.filterwarnings("ignore")

L.seed_everything(42)

n_epochs = 50
patience = 15

missing = 6
user = 0

batch_size = 512

log_save_dir = os.path.join(".", "results")
os.makedirs(log_save_dir, exist_ok=True)

# train_model_class = [LSTMModel, LSTMAttentionModel, FixedLSTMModel3, FixedLSTMModel3Attention1, FixedLSTMModel3_GAttention1, FixedLSTMModel3_GAttention1_Mulpath]
train_model_class = []
pc = get_PC()
if pc == 0:
    train_model_class = [LSTMModel, LSTMAttentionModel]
elif pc == 1:
    train_model_class = [FixedLSTMModel3, FixedLSTMModel3Attention1]
elif pc == 3:
    train_model_class = [FixedLSTMModel3_GAttention1, FixedLSTMModel3_GAttention1_Mulpath]
        
for model_class in train_model_class: 
    
    model_name = model_class.__name__
    print("Running for model", model_name)
    
    for missing_sensor_numbers in [6]: ## Changed for 1 missing sensor
        for user in [0,1]: ## Changed for user 2 only
            log_save_name = f"{model_name}/{missing_sensor_numbers}_missing/user{user}"
            
            data_module = DataModule(
                test_user=user, 
                missing_sensor_numbers=missing_sensor_numbers,
                batch_size=batch_size)
            
            net = model_class(input_size=42,
                              output_size=10,
                              sequence_length=256,
                              )
                        
            model_summary = ModelSummary(net, max_depth=6)
            print(model_summary)
                
            start_timer = time.perf_counter()
            print(f"\n*************training on User{user}*************")
            
            tensorboard_logger = TensorBoardLogger(save_dir=log_save_dir, name=log_save_name,)
            checkpoint_callback = ModelCheckpoint(
                dirpath=None,
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                filename="sample_{epoch:02d}-{step:02d}-{val_loss:02f}"
            )
            
            trainer = L.Trainer(
                logger=[tensorboard_logger],
                callbacks=[EarlyStopping(monitor="val_loss", patience=patience), checkpoint_callback],
                max_epochs=n_epochs,
                check_val_every_n_epoch=1,
                accelerator="gpu", 
                )

            trainer.fit(model=net, datamodule=data_module)
            trainer_test_dict = trainer.logged_metrics

            trainer.test(model=net, datamodule=data_module)
            trainer_test_dict.update(trainer.logged_metrics)

            print("trainer.logger.log_dir", trainer.logger.log_dir)

            for key in trainer_test_dict.keys():
                trainer_test_dict[key] = trainer_test_dict[key].item()

            with open(os.path.join(trainer.logger.log_dir, "result.json"), "w") as f:
                json.dump(trainer_test_dict, f, indent=4)
    
            end_timer = time.perf_counter()
            exec_time = end_timer - start_timer
            print(f"\n*************End training on User{user}*************")
            print(f"exec time:", exec_time)