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
import glob

sys.path.insert(0, os.path.join(os.path.abspath(''), "../.."))
from model.builder import LSTMModel, LSTMAttentionModel, ConvLSTMModel, ConvLSTMAttentionModel
from datamodule.datamodule import DataModule, FFTDataModule

from pc_control.pc_control import get_PC

import warnings
warnings.filterwarnings("ignore")

L.seed_everything(42)


model_class = ConvLSTMModel
model_name = model_class.__name__
result_dict = dict()
for missing_sensor_numbers in range(7):
    all_user_res = []
    for user in [0,1,2]:
        log_save_dir = os.path.join("../data/compgan_dataset/", "results", "82_classify_CV")
        log_save_name = f"{model_name}/{missing_sensor_numbers}_missing/user{user}"
        ckpt = glob.glob(os.path.join(log_save_dir, log_save_name, "version_0", "checkpoints", "*.ckpt"))[0]

        data_module = DataModule(
            test_user=user, 
            missing_sensor_numbers=missing_sensor_numbers,
            batch_size=16,
            test_mode=True)

        net = model_class.load_from_checkpoint(ckpt)
                    
        trainer = L.Trainer(
            accelerator="gpu", 
            )

        trainer.test(model=net, datamodule=data_module)
        print("logged_metrics", trainer.logged_metrics)
        all_user_res.append(trainer.logged_metrics["test_acc"].item())
    result_dict[missing_sensor_numbers] = sum(all_user_res)/16
print(result_dict)
with open(model_class.__name__ + ".json", "w") as f:
    json.dump(result_dict, f)
