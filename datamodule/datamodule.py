import os
import glob
import gc
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader

import lightning as L

import numpy  as np
from sklearn.preprocessing import StandardScaler

from . import *
from .dataset import CustomTrainDataset
sys.path.insert(0, os.path.join(os.path.abspath(''), ".."))

from utils.FFT import stft

curr = os.path.dirname(__file__)
DATA_ROOT = os.path.join(curr, "../data/compgan_dataset/")

train_data_file_name_ = "train_data{}.npy"
train_label_file_name_ = "train_label{}.npy"
test_data_file_name_ = "test_data{}.npy"
test_label_file_name_ = "test_label{}.npy"

TRAIN_FOLDER_PATH = os.path.join(DATA_ROOT, "train")
TEST_FOLDER_PATH = os.path.join(DATA_ROOT, "test")
RESULT_FOLDER_PATH = os.path.join(DATA_ROOT, "results")

assert os.path.isdir(TRAIN_FOLDER_PATH) and os.path.isdir(TEST_FOLDER_PATH)
os.makedirs(RESULT_FOLDER_PATH, exist_ok=True)

data_files = sorted(glob.glob(os.path.join(TRAIN_FOLDER_PATH, train_data_file_name_.format("*"))))

class DataModule(L.LightningDataModule):
    STANDARDIZE = False
    
    def __init__(self, test_user, missing_sensor_numbers, batch_size=2048, test_mode=False):
        super().__init__()
        self.save_hyperparameters()
        self.test_user = test_user
        self.missing_sensor_numbers = missing_sensor_numbers
        self.batch_size = batch_size
        self.scaler = None
        self.mode = CustomTrainDataset.TEST_MODE if test_mode else CustomTrainDataset.TRAIN_MODE

    def load_data(self, mode):
        if mode == "train":
            folder_path = TRAIN_FOLDER_PATH
            data_file_name = train_data_file_name_
            label_file_name = train_label_file_name_

            train_data_file_name = data_file_name.format(self.test_user)
            train_label_file_name = label_file_name.format(self.test_user)
    
            train_data_file_path = os.path.join(folder_path, train_data_file_name)
            train_label_file_path = os.path.join(folder_path, train_label_file_name)
            train_val_data, train_val_label = np.load(train_data_file_path), np.load(train_label_file_path)
            l, s, d, w = train_val_data.shape
            # self.scaler = StandardScaler()
            # train_val_data = self.scaler.fit_transform(train_val_data.reshape(l, -1)).reshape(l, s, d, w)
            # train_val_data = train_val_data.reshape(l, s ,d, w).transpose(0, 3, 1, 2)
            train_val_data = train_val_data.reshape(l, s * d, w)
            
        elif mode == "test":
            folder_path = TEST_FOLDER_PATH
            data_file_name = test_data_file_name_
            label_file_name = test_label_file_name_
    
            train_data_file_name = data_file_name.format(self.test_user)
            train_label_file_name = label_file_name.format(self.test_user)
    
            train_data_file_path = os.path.join(folder_path, train_data_file_name)
            train_label_file_path = os.path.join(folder_path, train_label_file_name)
            train_val_data, train_val_label = np.load(train_data_file_path), np.load(train_label_file_path)
            l, s, d, w = train_val_data.shape
            # train_val_data = self.scaler.transform(train_val_data.reshape(l, -1)).reshape(l, s, d, w)
            train_val_data = train_val_data.reshape(l, s * d, w)
    
        return train_val_data, train_val_label

    def setup(self, stage: str):
        # Assign Train/val split(s) for use in Dataloaders
    
        if stage == "validate" or stage == "fit":
            train_val_data, train_val_label = self.load_data("train")
            self.train_data, self.val_data, self.train_label, self.val_label = train_test_split(
                train_val_data, train_val_label, test_size=0.2, train_size=0.8, random_state=42, shuffle=True)
            
            assert self.mode == CustomTrainDataset.TRAIN_MODE
            
            self.train_dataset = CustomTrainDataset(
                self.mode, self.train_data, self.train_label, missing_sensor_numbers=self.missing_sensor_numbers)
            self.val_dataset = CustomTrainDataset(
                self.mode, self.val_data, self.val_label, missing_sensor_numbers=self.missing_sensor_numbers)

        elif stage == "test" or stage == "predict":
            train_val_data, train_val_label = self.load_data("test")
            self.test_data = train_val_data
            self.test_label = train_val_label

            self.test_dataset = CustomTrainDataset(
                self.mode, self.test_data, self.test_label, missing_sensor_numbers=self.missing_sensor_numbers)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,  num_workers=4, shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,  num_workers=4, shuffle=False, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,  num_workers=4, shuffle=False, pin_memory=True)

    def teardown(self, stage):
        print("teardown")
        if stage == "validate" or stage == "fit":
            del self.train_data, self.train_label
            del self.val_data, self.val_label
            del self.train_dataset
            del self.val_dataset
            
        elif stage == "test" or stage == "predict":
            del self.test_data, self.test_label
            del self.test_dataset
        gc.collect()



class FFTDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def load_data(self, mode):
        if mode == "train":
            folder_path = TRAIN_FOLDER_PATH
            data_file_name = train_data_file_name_
            label_file_name = train_label_file_name_

            train_data_file_name = data_file_name.format(self.test_user)
            train_label_file_name = label_file_name.format(self.test_user)
    
            train_data_file_path = os.path.join(folder_path, train_data_file_name)
            train_label_file_path = os.path.join(folder_path, train_label_file_name)
            train_val_data, train_val_label = np.load(train_data_file_path), np.load(train_label_file_path)
            l, s, d, w = train_val_data.shape
            train_val_data_reshape = train_val_data.reshape(l, s * d, w)
            
            all_channel_data = []
            for channel in range(train_val_data_reshape.shape[1]):
                one_channel_data = train_val_data_reshape[:,channel,:]
                _, one_channel_fft = stft(one_channel_data)
                # print(one_channel_fft.shape)
                all_channel_data.append(one_channel_fft.T)
                
            all_channel_data_concat = np.stack(all_channel_data).transpose(1,0,2)
            
            train_val_data = all_channel_data_concat
            
        elif mode == "test":
            folder_path = TEST_FOLDER_PATH
            data_file_name = test_data_file_name_
            label_file_name = test_label_file_name_
    
            train_data_file_name = data_file_name.format(self.test_user)
            train_label_file_name = label_file_name.format(self.test_user)
    
            train_data_file_path = os.path.join(folder_path, train_data_file_name)
            train_label_file_path = os.path.join(folder_path, train_label_file_name)
            train_val_data, train_val_label = np.load(train_data_file_path), np.load(train_label_file_path)
            l, s, d, w = train_val_data.shape
            
            train_val_data_reshape = train_val_data.reshape(l, s * d, w)
            
            all_channel_data = []
            for channel in range(train_val_data_reshape.shape[1]):
                one_channel_data = train_val_data_reshape[:,channel,:]
                _, one_channel_fft = stft(one_channel_data)
                # print(one_channel_fft.shape)
                all_channel_data.append(one_channel_fft.T)
                
            all_channel_data_concat = np.stack(all_channel_data).transpose(1,0,2)
            
            train_val_data = all_channel_data_concat
    
        return train_val_data, train_val_label
    

class FFTDataModule_1_sensor(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def load_data(self, mode):
        if mode == "train":
            folder_path = TRAIN_FOLDER_PATH
            data_file_name = train_data_file_name_
            label_file_name = train_label_file_name_

            train_data_file_name = data_file_name.format(self.test_user)
            train_label_file_name = label_file_name.format(self.test_user)
    
            train_data_file_path = os.path.join(folder_path, train_data_file_name)
            train_label_file_path = os.path.join(folder_path, train_label_file_name)
            train_val_data, train_val_label = np.load(train_data_file_path), np.load(train_label_file_path)
            l, s, d, w = train_val_data.shape
            train_val_data_reshape = train_val_data.reshape(l, s * d, w)
            
            all_channel_data = []
            for channel in range(train_val_data_reshape.shape[1]):
                one_channel_data = train_val_data_reshape[:,channel,:]
                _, one_channel_fft = stft(one_channel_data)
                # print(one_channel_fft.shape)
                all_channel_data.append(one_channel_fft.T)
                
            all_channel_data_concat = np.stack(all_channel_data).transpose(1,0,2)
            
            train_val_data = all_channel_data_concat
            
        elif mode == "test":
            folder_path = TEST_FOLDER_PATH
            data_file_name = test_data_file_name_
            label_file_name = test_label_file_name_
    
            train_data_file_name = data_file_name.format(self.test_user)
            train_label_file_name = label_file_name.format(self.test_user)
    
            train_data_file_path = os.path.join(folder_path, train_data_file_name)
            train_label_file_path = os.path.join(folder_path, train_label_file_name)
            train_val_data, train_val_label = np.load(train_data_file_path), np.load(train_label_file_path)
            l, s, d, w = train_val_data.shape
            
            train_val_data_reshape = train_val_data.reshape(l, s * d, w)
            
            all_channel_data = []
            for channel in range(train_val_data_reshape.shape[1]):
                one_channel_data = train_val_data_reshape[:,channel,:]
                _, one_channel_fft = stft(one_channel_data)
                # print(one_channel_fft.shape)
                all_channel_data.append(one_channel_fft.T)
                
            all_channel_data_concat = np.stack(all_channel_data).transpose(1,0,2)
            
            train_val_data = all_channel_data_concat
    
        return train_val_data, train_val_label
    
    def setup(self, stage: str):
        # Assign Train/val split(s) for use in Dataloaders
        assert self.mode == CustomTrainDataset.TEST_MODE
    
        if stage == "validate" or stage == "fit":
            train_val_data, train_val_label = self.load_data("train")
            self.train_data, self.val_data, self.train_label, self.val_label = train_test_split(
                train_val_data, train_val_label, test_size=0.2, train_size=0.8, random_state=42, shuffle=True)
            
            # TRAIN EXCEPT MISSING FROM 0 -> MISSING_NUMBER - 1
            self.train_dataset = CustomTrainDataset(
                self.mode, self.train_data, self.train_label, missing_sensor_numbers=self.missing_sensor_numbers)
            self.val_dataset = CustomTrainDataset(
                self.mode, self.val_data, self.val_label, missing_sensor_numbers=self.missing_sensor_numbers)

        elif stage == "test" or stage == "predict":
            train_val_data, train_val_label = self.load_data("test")
            self.test_data = train_val_data
            self.test_label = train_val_label

            self.test_dataset = CustomTrainDataset(
                self.mode, self.test_data, self.test_label, missing_sensor_numbers=self.missing_sensor_numbers)
