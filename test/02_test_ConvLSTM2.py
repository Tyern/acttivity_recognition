from functools import reduce
import numpy as np
import torch
from torch import nn

import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import ConvLSTM2D
# from tensorflow.keras.models import Input, Dense, Flatten, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, ConvLSTM2D, Reshape

import sys
from pathlib import Path
curr = Path(__file__).resolve()
sys.path.insert(0, curr.parent.parent.as_posix())
from model.builder import ConvLSTMCell, ConvLSTM, BaseModel

def build_tf_cnn_lstm():
    inputs=tf.keras.layers.Input(shape=(256,42,1))
    
    x=tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',strides=1,padding='same')(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    #x=Dropout(0.5)(x)
    x=tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu',strides=1,padding='same')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    #x=Dropout(0.5)(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    #x = Flatten()(x)
    x=tf.keras.layers.Reshape((1,128,21,64))(x)
    x=ConvLSTM2D(filters=64,kernel_size=(1,3),activation='relu', padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(64, activation='relu')(x)
    y=tf.keras.layers.Dense(10, activation='softmax')(x)
    model=tf.keras.models.Model(inputs=inputs,outputs=y)
    return model

class ConvLSTMModel(BaseModel):
    def __init__(
        self, 
        hidden_size=64, 
        sequence_length=256, 
        cnn_filter_size=64, 
        input_size=42,
        output_size=10, 
        **kwargs):

        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(10, input_size, sequence_length)
        
        self.conv1 = nn.Conv2d(1, cnn_filter_size//2, (3, 3), 1, "same")
        self.bn1 = nn.BatchNorm2d(cnn_filter_size//2)
        self.conv2 = nn.Conv2d(cnn_filter_size//2, cnn_filter_size, (3, 3), 1, "same")
        self.bn2 = nn.BatchNorm2d(cnn_filter_size)
        self.mp = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_lstm = ConvLSTM(input_dim=(cnn_filter_size), hidden_dim=hidden_size, kernel_size=(1,3), num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(172032, 64)
        self.fc2 = nn.Linear(64, output_size)
        
        # Activation
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        
        out = self.conv1(x)
        out = self.activation(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.bn2(out)
        out = self.mp(out)
        
        out = out.view(x.shape[0], 1, 64, 128, 21)
        _, last_state_list = self.conv_lstm(out)
        out = last_state_list[-1][0] # last state ht
        out = self.activation(out)
        
        out = out.view(x.shape[0], -1)
        out = self.fc1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        out = self.softmax(out)
        return out
        
if __name__ == "__main__":
    tf_model = build_tf_cnn_lstm()
    print(tf_model.summary())
    
    torch_model = ConvLSTMModel()
    print(ModelSummary(torch_model))


