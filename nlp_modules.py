# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 08:37:20 2022

@author: Khuru
"""


from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Bidirectional,Embedding, Masking
from tensorflow.keras.models import Sequential
import numpy as np

#%%
class ModelCreation():
    def _init_(self):
        pass
    
    def simple_lstm_layer(self,drop_rate=0.2,embedding_dim = 150,
                          vocab_size = 1000,output_node=5):

        model = Sequential()
        model.add(Input(shape=(333))) #np.shape(X_train)[1:]
        model.add(Embedding(vocab_size,embedding_dim))
        model.add(Bidirectional(LSTM(128,return_sequences=(True))))
        model.add(Dropout(drop_rate))
        model.add(Masking(mask_value=0)) #Masking layer is to remove the 0 from padded data
                                         #- replace the 0 with the data values
        model.add(Bidirectional(LSTM(128,return_sequences=(True))))
        model.add(Dropout(drop_rate))
        model.add(LSTM(128))
        model.add(Dropout(drop_rate))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node, activation='softmax'))
        model.summary()
        
        return model