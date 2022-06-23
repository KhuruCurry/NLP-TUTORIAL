# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:12:05 2022

@author: Khuru
"""

import os
import re
import json
import pickle  
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding,Bidirectional,Masking


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



#%% Data Loading
CSV_URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_category.json')
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')

log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)
#%%

vocab_size = 1000 
oov_token = 'OOV'
max_len = 333

# EDA
# Step 1) Data Loading
df = pd.read_csv(CSV_URL)

# Step 2) Data Inspection
df.head(10) #  visualise first 10 rows
df.tail(10) #  visualise last 10 rows
df.info()

df['category'].unique()
df['category'][5]
df['text'][5]

df.duplicated().sum() # to check for duplicated data
df[df.duplicated()] # to visualise the duplicated data

# Step 3) Data Cleaning
# to remove duplicated data
df = df.drop_duplicates() # to remove duplicates
print(df)

text = df['text'].values # Features : X
category = df['category'].values # Features : y

for index,rev in enumerate(text):

    text[index] = re.sub('<.*?>',' ',rev) 

    # convert into lower case 
    # remove numbers
    # ^ means NOT
    text[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()

# Step 4) Features selection
# Nothing to select

#%% Data preprocessing

#           1) Convert into lower case
#           2) Tokenization
# Tokenization = splitting up a larger body of text into smaller lines, words

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(text) # learn all of the words
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(text) # to convert into numbers

#3) Padding & truncating
length_of_text = [len(i) for i in train_sequences] # list comprehension
print(np.median(length_of_text)) # to get the number of max length for padding

#padded sequences is to have equal length of data and have their own numbers
#check at padded text (variable explorer)
padded_text = pad_sequences(train_sequences,maxlen=max_len,
                            padding='post',
                            truncating='post')

#One Hot Encoding for the target
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))


#         5) Train test split

X_train,X_test,y_train,y_test = train_test_split(padded_text,
                                                 category,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model development
# USE LSTM
#achive more than 70% f1 score

# for bidirectional
embedding_dim = 150

model = Sequential()
model.add(Input(shape=(333))) #np.shape(X_train)[1:]
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(128,return_sequences=(True))))
model.add(Dropout(0.2))
model.add(Masking(mask_value=0)) #Masking layer is to remove the 0 from padded data
                                 #- replace the 0 with the data values
model.add(Bidirectional(LSTM(128,return_sequences=(True))))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(np.shape(category)[1], activation='softmax'))
model.summary()


plot_model(model)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

#callbacks
tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)
early_stopping_callback = EarlyStopping(monitor='loss',patience=5)

hist = model.fit(X_train,y_train,
                 epochs=3,
                 batch_size=20,
                 validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])

                 

#%% Plot
hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--',label='Training acc')
plt.plot(hist.history['val_acc'],label='Validation acc')
plt.legend()
plt.show()

#%% Model evaluation

y_true = y_test
y_pred = model.predict(X_test)
#%%

# =============================================================================
# print(classification_report(y_true,y_pred))
# print(accuracy_score(y_true,y_pred))
# print(confusion_matrix(y_true,y_pred))
# =============================================================================

#%% Model saving


model.save(MODEL_SAVE_PATH)

token_json = tokenizer.to_json()

with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)


