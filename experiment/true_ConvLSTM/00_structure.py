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

sys.path.insert(0, os.path.join(os.path.abspath(''), "../.."))
from model.builder import LSTMModel, LSTMAttentionModel, ConvLSTMModel, ConvLSTMAttentionModel
from datamodule.datamodule import DataModule, FFTDataModule

from pc_control.pc_control import get_PC

import warnings
warnings.filterwarnings("ignore")

L.seed_everything(42)


if __name__ == "__main__":
    model_class_list = [ConvLSTMModel]
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
        