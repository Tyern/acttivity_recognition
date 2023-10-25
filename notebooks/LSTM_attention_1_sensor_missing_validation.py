#!/usr/bin/env python
# coding: utf-8

# In[56]:


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


# In[57]:


DATA_ROOT = "../data/SensorData/"
user_file_prefix = "User"

TRAIN_FOLDER_PATH = os.path.join(DATA_ROOT, "train")
TEST_FOLDER_PATH = os.path.join(DATA_ROOT, "test")
RESULT_FOLDER_PATH = os.path.join(DATA_ROOT, "results")

os.makedirs(TRAIN_FOLDER_PATH, exist_ok=True)
os.makedirs(TEST_FOLDER_PATH, exist_ok=True)

data_files = sorted(glob.glob(os.path.join(DATA_ROOT, f"{user_file_prefix}*.csv")))
print(data_files)


# ## Test with User00.csv

# In[32]:


def segment(data_df, label_list, Window_size = 200, over_lap = 0.5, margin = 200):
    data_df_index_list = []
    index_label_list = []
    
    overlap_data = Window_size * over_lap

    index = 0
    loop = 0
    current_label = None
    
    while index < data_df.shape[0]:
        if index + Window_size >= data_df.shape[0]: break
    
        if current_label is not None and data_df['label'][index] == current_label:
            index += 1
            continue
            
        if pd.isna(data_df['label'][index]):
            index = index + 1
            continue

        count = 0
        while count < margin:
            count, index = count + 1, index + 1

        if index + Window_size >= data_df.shape[0]: break
        current_label = data_df.loc[index, 'label']

        while not pd.isna(data_df['label'][index + Window_size]) and data_df['label'][index + Window_size] == current_label:
#                 signal_data = data_df.iloc[index: index + Window_size, :]

            # Each data will be at size column(6 x 5 = 30) x Window_size
            # achieved by `data_df.loc[index: index + self.Window_size, :]` for index in data_df_index_list.
            # Data label will be `data_df.loc[index + self.Window_size, "label"]` for index in data_df_index_list.
            data_df_index_list.append(index)
            index = index + Window_size

            # add label of the last row of sequence
            index_label_list.append(data_df['label'][index])

            index = index - int(overlap_data)
            if index + Window_size >= data_df.shape[0]: break
                
    return data_df_index_list, index_label_list


# In[33]:


data_df = pd.read_csv(data_files[0])
data_df


# In[34]:


data_df["label"].value_counts()


# In[58]:


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


# In[59]:


# important
eng_label_dict = dict(zip(
    label_list,
    ['Walking', 'Upstair', 'Bed_Standup', 'Change_Bed', 'Change_Bed_Standup', 'Sit_Down', 'Sit', 'Stand', 'Shoulder_Exercise', 'Hip_Exercise']
))

eng_label_list = [eng_label_dict[i] for i in label_list]


# In[37]:


window_size = 256
data_df_index_list, index_label_list = segment(data_df, label_list, Window_size=window_size, over_lap=3/4)


# In[38]:


data_list = []
for index, label_number in zip(data_df_index_list, index_label_list):
    x = data_df.iloc[index: index + window_size, :].drop(["label"], axis=1).values
    data_list.append(x)
data_list = np.array(data_list)


# In[39]:


data_list.shape


# In[40]:


sc = StandardScaler()

data_num, window_size, feature_num = data_list.shape
features_reshape = data_list.reshape(-1, feature_num)
features_norm = sc.fit_transform(features_reshape)

# convert back t feature size
features = features_norm.reshape(data_num, window_size, feature_num)


# In[ ]:





# ## create function to load feature from 1 csv file

# In[41]:


def load_csv_data(csv_file_path):
    window_size = 256
    data_df = pd.read_csv(csv_file_path)
    data_df_index_list, index_label_list = segment(data_df, label_list, Window_size=window_size, over_lap=3/4)
    
    data_list = []
    for index, label_number in zip(data_df_index_list, index_label_list):
        x = data_df.iloc[index: index + window_size, :].drop(["label"], axis=1).values
        data_list.append(x)
    data_list = np.array(data_list)
    
    sc = StandardScaler()

    data_num, window_size, feature_num = data_list.shape
    features_reshape = data_list.reshape(-1, feature_num)
    features_norm = sc.fit_transform(features_reshape)

    # convert back t feature size
    features = features_norm.reshape(data_num, window_size, feature_num)
    
    return features, index_label_list


# ## create LOSO train, test dataset from 17 users

# In[42]:


temp_dict = {}

for index, csv_file_path in enumerate(data_files):
    
    identifier = os.path.splitext(os.path.basename(csv_file_path))[0]
    features, index_label_list = load_csv_data(csv_file_path)
    
    temp_dict[identifier] = (features, index_label_list)


# In[43]:


temp_dict_keys = sorted(temp_dict.keys())

for test_identifier in temp_dict_keys:
    
    train_features_list = []
    train_label_list = []

    for identifier in temp_dict_keys:
        
        if test_identifier == identifier:
            # test_set
            features, index_label_list = temp_dict[identifier]
            np.save(os.path.join(TEST_FOLDER_PATH, test_identifier), features)
            np.save(os.path.join(TEST_FOLDER_PATH, f"{test_identifier}_label"), index_label_list)

        else:
            # train_set
            train_features_list.append(temp_dict[identifier][0])
            train_label_list.append(temp_dict[identifier][1])

    train_features = np.concatenate(train_features_list, axis=0)
    train_label = np.concatenate(train_label_list, axis=0)

    np.save(os.path.join(TRAIN_FOLDER_PATH, test_identifier), train_features)
    np.save(os.path.join(TRAIN_FOLDER_PATH, f"{test_identifier}_label"), train_label)
        


# ## Dataset, dataloader setup

# In[44]:


save_filename_list = [(f"{identifier}.npy", f"{identifier}_label.npy") for identifier in temp_dict_keys]
save_filename_list


# In[45]:


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, feature_file, label_file, missing_sensor_id_list=None):
        self.features = np.load(feature_file)
        self.label = np.load(label_file)
        
        if missing_sensor_id_list is not None:
            for missing_sensor_id in missing_sensor_id_list:
                self.features[:, :, missing_sensor_id*6:(missing_sensor_id+1)*6] = 0
        
        assert len(self.features) == len(self.label), "features len is not equal to label len"
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        label = self.label[idx]
        return x, label


# In[46]:


# test
dataset = CustomDataset(
    os.path.join(TRAIN_FOLDER_PATH, save_filename_list[0][0]),
    os.path.join(TRAIN_FOLDER_PATH, save_filename_list[0][1]),
    missing_sensor_id_list=[0]
)


# In[47]:


dataset[0]


# In[48]:


# test
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


# In[49]:


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


# ## Model definition

# In[50]:


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


# In[51]:


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
        
        y_pred = self.forward(X)
        # 2. Calculate  and accumulate loss
        loss = F.cross_entropy(y_pred, y)
        
        self.log("val_loss", loss, prog_bar=True)
        


# In[52]:


model = LSTMModel()
summary(model)


# In[53]:


# save_filename_list = [save_filename_list[0]]


# In[54]:


print(list(combinations(range(7),2)))


# In[55]:


# save_filename_list = [save_filename_list[0]]
# print(f"only use file: {save_filename_list}")


# In[34]:


from tqdm.auto import tqdm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

batch_size = 1024
patience = 20

all_test_pred = {}
missing_sensor_numbers = 3
print(f"start learning missing_sensor_numbers = {missing_sensor_numbers}")

for missing_index in range(1):

    all_test = []
    all_pred = []

    # kfold_train_test_index_list = [kfold_train_test_index_list[0]]

    for i, save_filename_feature_label in enumerate(save_filename_list):
        print(f"\n*************{save_filename_feature_label[0]}*************")
        dataset = CustomDataset(
            os.path.join(TRAIN_FOLDER_PATH, save_filename_feature_label[0]),
            os.path.join(TRAIN_FOLDER_PATH, save_filename_feature_label[1]),
            missing_sensor_id_list=None,
        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        test_dataset =  CustomDataset(
            os.path.join(TEST_FOLDER_PATH, save_filename_feature_label[0]),
            os.path.join(TEST_FOLDER_PATH, save_filename_feature_label[1]),
            missing_sensor_id_list=None,
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
with open(os.path.join(RESULT_FOLDER_PATH, "all_test_pred.pkl"), "wb") as f:
    pickle.dump(all_test_pred, f)


# In[60]:


os.stat(os.path.join(RESULT_FOLDER_PATH, "all_test_pred.pkl"))


# In[61]:


with open(os.path.join(RESULT_FOLDER_PATH, "all_test_pred.pkl"), "rb") as f:
    all_test_pred = pickle.load(f)


# In[67]:


missing_index = 0

all_tall_test, all_pred = all_test_pred[missing_index]

# print("missing index", missing_index)
all_test_with_label = [label_list[i] for i in all_test]
all_pred_with_label = [label_list[i] for i in all_pred]

cf = confusion_matrix(all_test_with_label, all_pred_with_label, labels=label_list)
sns.heatmap(cf, annot=True, xticklabels=eng_label_list, yticklabels=eng_label_list, fmt='g')


# In[65]:


# missing_index = 2
row_item = 1

fig, ax = plt.subplots(len(all_test_pred.keys()) // row_item + len(all_test_pred.keys()) % row_item, row_item, figsize=(13,28))

for idx, missing_index in enumerate(all_test_pred.keys()):
    all_test, all_pred = all_test_pred[missing_index]
    
    # print("missing index", missing_index)
    all_test_with_label = [label_list[i] for i in all_test]
    all_pred_with_label = [label_list[i] for i in all_pred]
    
    cf = confusion_matrix(all_test_with_label, all_pred_with_label, labels=label_list)
    sns.heatmap(cf, ax=ax[idx//row_item][idx%row_item], annot=True, xticklabels=eng_label_list, yticklabels=eng_label_list, fmt='g')
    ax[idx//row_item][idx%row_item].set_title(f"missing_index {missing_index} confusion matrix")
    
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER_PATH, "all_test_pred.jpg"))


# In[ ]:





# In[ ]:


# print("precision_recall_fscore_support: ")
# print()
# print(*eng_label_list, sep=" "*4)
# print(*precision_recall_fscore_support(all_test_with_label, all_pred_with_label, labels=label_list), sep="\n")


# In[ ]:


all_test, all_pred = all_test_pred[missing_index]
print(all_test[0].cpu().item())


# In[23]:


from sklearn.metrics import accuracy_score

for missing_index, (all_test, all_pred) in all_test_pred.items():
    
    all_test = list(map(lambda x: x.cpu().item(), all_test))
    all_pred = list(map(lambda x: x.cpu().item(), all_pred))
    
    print("missing_index", missing_index,":", accuracy_score(all_test, all_pred))
    # print(accuracy_score(all_test, all_pred))
    # print()


# In[ ]:




