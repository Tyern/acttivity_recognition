#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torchinfo import summary

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy  as np

import json
import os
import glob
import pickle
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")


# In[2]:


DATA_ROOT = "../data/compgan_dataset/"

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
print(data_files)
print(len(data_files))


# In[3]:


# important

label_list = ['歩行(平地)',
 '歩行(階段)',
 'ベッド上での起き上がり',
 'ベッド椅子間の乗り移り(立つ)',
 'ベッド椅子間の乗り移り(立たない)',
 '立ち座り',
 '座位保持・座位バランス',
 '立位保持・立位バランス',
 '関節可動域増大訓練(肩)',
 '関節可動域増大訓練(股関節)']

label_dict = dict(enumerate(label_list))


# In[4]:


# important
eng_label_dict = dict(zip(
    label_list,
    ['Walking', 'Upstair', 'Bed_Standup', 'Change_Bed', 'Change_Bed_Standup', 'Sit_Down', 'Sit', 'Stand', 'Shoulder_Exercise', 'Hip_Exercise']
))

eng_label_list = [eng_label_dict[i] for i in label_list]


# In[5]:


test_user = 0


# In[6]:


def load_train_test(test_user):
    train_data_file_name = train_data_file_name_.format(test_user)
    train_label_file_name = train_label_file_name_.format(test_user)
    test_data_file_name = test_data_file_name_.format(test_user)
    test_label_file_name = test_label_file_name_.format(test_user)

    train_data_file_path = os.path.join(TRAIN_FOLDER_PATH, train_data_file_name)
    train_label_file_path = os.path.join(TRAIN_FOLDER_PATH, train_label_file_name)
    test_data_file_path = os.path.join(TEST_FOLDER_PATH, test_data_file_name)
    test_label_file_path = os.path.join(TEST_FOLDER_PATH, test_label_file_name)

    train_data, train_label = np.load(train_data_file_path), np.load(train_label_file_path)
    test_data, test_label = np.load(test_data_file_path), np.load(test_label_file_path)

    return train_data, train_label, test_data, test_label


# In[7]:


# test
train_data, train_label, test_data, test_label = load_train_test(test_user)


# In[8]:


print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)


# In[9]:


# test
l, s, d, w = train_data.shape
train_data = train_data.reshape(l, s * d, w)
train_data = train_data.transpose(0,2,1)

train_data.shape


# In[10]:


feature_num = s * d


# In[11]:


class CustomDataset(Dataset):
    def __init__(self, feature_data, label_data, missing_sensor_id_list=None):
        l, s, d, w = feature_data.shape
        self.features = feature_data.reshape(l, s * d, w).transpose(0,2,1)
        self.label = label_data
        
        if missing_sensor_id_list is not None:
            for missing_sensor_id in missing_sensor_id_list:
                self.features[:, :, missing_sensor_id*6:(missing_sensor_id+1)*6] = 0
        
        assert len(self.features) == len(self.label), "features len is not equal to label len"
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        label = self.label[idx]
        return x, int(label)


# In[12]:


# test
train_data, train_label, test_data, test_label = load_train_test(test_user)
dataset = CustomDataset(
    train_data,
    train_label,
    missing_sensor_id_list=[0]
)


# In[13]:


dataset[0]


# In[14]:


# test
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


# In[15]:


# test
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=8,
    num_workers=0, # number of subprocesses to use for data loading
    shuffle=True)

val_dataloader = DataLoader(
    train_dataset, 
    batch_size=8,
    num_workers=0, # number of subprocesses to use for data loading
    shuffle=False)


# In[16]:


next(train_dataloader.__iter__())[0].shape


# ## model definition

# In[17]:


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


# In[22]:


class LSTMModel(pl.LightningModule):
    def __init__(self, hidden_size=128, input_size=30, output_size=6):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(1024, 200, input_size)
        
        self.rnn1 = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.attention1 = SelfAttention(
            input_dim=hidden_size)

        self.rnn2 = nn.LSTM(input_size=hidden_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.attention2 = SelfAttention(
            input_dim=hidden_size
        )
        
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
        
        self.all_test = []
        self.all_pred = []
        
    def forward(self, x):
        activation, _ = self.rnn1(x)
        activation = self.attention1(activation)
        activation, _ = self.rnn2(activation)
        activation = self.attention2(activation)

        b, _, _ = activation.size()
        
        lstm_output = activation[:,-1,:].view(b,-1)
        
        seq_1_output = self.seq_1(lstm_output)
        seq_2_output = self.seq_2(lstm_output)
        
        output = torch.concat([lstm_output, seq_1_output, seq_2_output], dim=1)
        output = self.classifier(output)
        
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.0005)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        X = X.float()
        # 1. Forward pass
        y_pred = self.forward(X)
        # 2. Calculate  and accumulate loss
        loss = F.cross_entropy(y_pred, y)
        
        self.log("train_loss", loss, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        X, y = batch
        X = X.float()
        y = y
    
        # 1. Forward pass
        test_pred_logits = self.forward(X)

        # Calculate and accumulate accuracy
        test_pred_labels = test_pred_logits.argmax(dim=1)
        test_acc = ((test_pred_labels == y).sum().item()/len(test_pred_labels))
        self.log("test_acc", test_acc)
        
        self.all_pred = test_pred_labels
        self.all_test = y
        
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, y = batch
        X = X.float()
        y = y
        
        y_pred = self.forward(X)
        # 2. Calculate  and accumulate loss
        loss = F.cross_entropy(y_pred, y)
        
        self.log("val_loss", loss, prog_bar=True)
        


# In[23]:


model = LSTMModel()
summary(model)


# In[24]:


list(combinations(range(7),1))


# In[25]:


from tqdm.auto import tqdm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

batch_size = 1024
patience = 20
# missing_sensor_numbers = 0
all_test_pred = {}

for missing_sensor_numbers in range(6):
    print(f"start learning missing_sensor_numbers = {missing_sensor_numbers}")
    
    for missing_index in combinations(range(7),missing_sensor_numbers):
    
        all_test = []
        all_pred = []
    
        # kfold_train_test_index_list = [kfold_train_test_index_list[0]]
    
        for i in range(len(data_files)):
            print(f"\n*************training on User{i}*************")
            train_data, train_label, test_data, test_label = load_train_test(i) # test user
            dataset = CustomDataset(
                train_data,
                train_label,
                missing_sensor_id_list=missing_index,
            )
    
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
    
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            test_dataset =  CustomDataset(
                test_data,
                test_label,
                missing_sensor_id_list=missing_index,
            )
    
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=batch_size,
                num_workers=4, # number of subprocesses to use for data loading
                shuffle=True)
    
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=batch_size,
                num_workers=2, # number of subprocesses to use for data loading
                shuffle=False)
    
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=2, # number of subprocesses to use for data loading
                shuffle=False)
    
            model = LSTMModel(hidden_size=128, input_size=feature_num, output_size=len(label_list))
    
            tb_logger = TensorBoardLogger(".")
    
            trainer = pl.Trainer(
                logger=tb_logger,
                callbacks=[EarlyStopping(monitor="val_loss", patience=patience, mode="min")],
            )
            trainer.fit(model, train_dataloader, val_dataloader)
            trainer.test(model, test_dataloader)
    
            all_test.extend(model.all_test)
            all_pred.extend(model.all_pred)
            
        all_test_pred[missing_index] = (all_test, all_pred)
    
    os.makedirs(RESULT_FOLDER_PATH, exist_ok=True)
    with open(os.path.join(RESULT_FOLDER_PATH, f"all_test_pred_{missing_sensor_numbers}.pkl"), "wb") as f:
        pickle.dump(all_test_pred, f)


# In[26]:


for missing_sensor_numbers in range(6):
    with open(os.path.join(RESULT_FOLDER_PATH, f"all_test_pred_{missing_sensor_numbers}.pkl"), "rb") as f:
        all_test_pred = pickle.load(f)

    if all_test_pred == 0:
        missing_index = list(all_test_pred.keys())[0]
        
        all_tall_test, all_pred = all_test_pred[missing_index]
        
        # print("missing index", missing_index)
        all_test_with_label = [label_list[i] for i in all_test]
        all_pred_with_label = [label_list[i] for i in all_pred]
        
        cf = confusion_matrix(all_test_with_label, all_pred_with_label, labels=label_list)
        sns.heatmap(cf, annot=True, xticklabels=eng_label_list, yticklabels=eng_label_list, fmt='g')
    else:
        row_item = 2
        fig, ax = plt.subplots(len(all_test_pred.keys()) // row_item + len(all_test_pred.keys()) % row_item, row_item, figsize=(13,28))
        
        for idx, missing_index in enumerate(all_test_pred.keys()):
            all_test, all_pred = all_test_pred[missing_index]
            
            # print("missing index", missing_index)
            all_test_with_label = [label_list[i] for i in all_test]
            all_pred_with_label = [label_list[i] for i in all_pred]
            
            cf = confusion_matrix(all_test_with_label, all_pred_with_label, labels=label_list)
            sns.heatmap(cf, ax=ax[idx//row_item][idx%row_item], annot=True, xticklabels=eng_label_list, yticklabels=eng_label_list, fmt='g')
            ax[idx//row_item][idx%row_item].set_title(f"missing_index {missing_index} confusion matrix")
            
    plt.savefig(os.path.join(RESULT_FOLDER_PATH, f"all_test_pred_{missing_sensor_numbers}.jpg"))


# In[ ]:




