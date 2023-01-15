import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
import math

train_df = pd.read_csv('train_pre.csv', sep='\t')
test_df = pd.read_csv('test_pre.csv', sep='\t')

X = train_df.dropna()
y = np.array(X['Correct First Attempt']).astype(int).ravel()
del X['Correct First Attempt']
XX = test_df.dropna()
yy = np.array(XX['Correct First Attempt']).astype(int).ravel()
del XX['Correct First Attempt']

scalar = MinMaxScaler(feature_range=(0,1))
X = scalar.fit_transform(X)
XX = scalar.fit_transform(XX)
"""
train_size = int(X.shape[0]*0.8)
test_size = X.shape[0]-train_size
X=np.array(X)
X_train,X_val =X[0:train_size,:],X[train_size:,:]

y_train=y[:train_size,]
y_val=y[train_size:,]
"""
X_train,y_train=X,y
X_val,y_val=XX,yy
#y=y.reshape(-1,1)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

filepath='weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, mOnitor='val_acc', verbose=1,save_best_Only=True,mode='max',period=2) 
callbacks_list = [checkpoint]

from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
model = Sequential()
model.add(GRU(1,input_shape=(1,12)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(X_train,y_train,epochs=1000,batch_size=4,verbose=2,callbacks=callbacks_list)

model.load_weights('weights.best.hdf5')
#model.compile(loss='mean_squared_error', optimizer='adam')
trainPredict = model.predict(X_train)
testPredict = model.predict(X_val)

trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
print("Train Sore :%.4f RMSE" % (trainScore))
testScore = math.sqrt(mean_squared_error(y_val, testPredict))
print("Test Sore :%.4f RMSE" % (testScore))
