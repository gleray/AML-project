import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import theano
import theano.tensor as T
import lasagne

os.chdir('/run/user/1000/gvfs/dav:host=data.deic.dk,ssl=true,user=gleray%40elektro.dtu.dk,prefix=%2Fremote.php%2Fwebdav/Data/energy_disaggregation')

# load the data from data.deic
data=pd.read_csv('REDD/low_frequency/house_1/channel_1.dat',header=None,sep=" ")
fridge=pd.read_csv('REDD/low_frequency/house_1/channel_5.dat',header=None,sep=" ")
microwave=pd.read_csv('REDD/low_frequency/house_1/channel_11.dat',header=None,sep=" ")
oven=pd.read_csv('REDD/low_frequency/house_1/channel_3.dat',header=None,sep=" ")

#change the timestep and the resolution
data[0] = pd.to_datetime(data[0], unit='s')
data = data.set_index(0).rename(columns={1: 'House1'})
dataRes=data.resample('10s',how='mean',limit=5,loffset="10s")

fridge[0] = pd.to_datetime(fridge[0], unit='s')
fridge = fridge.set_index(0).rename(columns={1: 'Fridge'})
fridgeRes=fridge.resample('10s',how='mean',limit=5)

microwave[0] = pd.to_datetime(microwave[0], unit='s')
microwave = microwave.set_index(0).rename(columns={1: 'Microwave'})
microwaveRes=microwave.resample('10s',how='mean',limit=5)

oven[0] = pd.to_datetime(oven[0], unit='s')
oven = oven.set_index(0).rename(columns={1: 'oven'})
ovenRes=oven.resample('10s',how='mean',limit=5)

# change the shape to get training samples of size 1000
MAX_LENGTH = 1000
maskMat=dataRes.notnull().astype(int)
mask=np.reshape(maskMat.values[:302000],(-1,MAX_LENGTH))

X=np.reshape(dataRes.values[:302000],(-1,MAX_LENGTH))
meanX = np.nan_to_num(X).mean(axis=1)
X-=meanX.reshape((meanX.shape[0]), 1)


ovenY=np.reshape(ovenRes.values[:302000],(-1,MAX_LENGTH))
meanovenY = np.nan_to_num(ovenY).mean(axis=1)
ovenY-=meanovenY.reshape((meanovenY.shape[0]), 1)


microwaveY=np.reshape(microwaveRes.values[:302000],(-1,MAX_LENGTH))
meanmicrowaveY = np.nan_to_num(microwaveY).mean(axis=1)
microwaveY-=meanmicrowaveY.reshape((meanmicrowaveY.shape[0]), 1)

fridgeY=np.reshape(fridgeRes.values[:302000],(-1,MAX_LENGTH))
meanfridgeY = np.nan_to_num(fridgeY).mean(axis=1)
fridgeY-=meanfridgeY.reshape((meanfridgeY.shape[0]), 1)

# remove the training set with missing data
maskDrop = np.ones(len(X), dtype=bool)
maskDrop[np.where(np.any(np.isnan(X),axis=1))[0]] = False

X=X[maskDrop,...]
fridgeY=fridgeY[maskDrop,...]
ovenY=ovenY[maskDrop,...]
microwaveY=microwaveY[maskDrop,...]
mask=mask[maskDrop,...]

#print (np.where(np.any(np.isnan(X),axis=1))[0])
#print (np.where(np.any(np.isnan(fridgeY),axis=1))[0])

# plt.plot(X[1,:])
# plt.plot(fridgeY[1,:])
# plt.plot(microwaveY[1,:])
# plt.plot(ovenY[1,:])
# plt.show()

# generate the test sets (evaluation set)
maskVal=np.reshape(maskMat.values[302000:310000],(-1,MAX_LENGTH))

XVal=np.reshape(dataRes.values[302000:310000],(-1,MAX_LENGTH))
meanXVal = np.nan_to_num(XVal).mean(axis=1)
XVal-=meanXVal.reshape((meanXVal.shape[0]), 1)

microwaveVal=np.reshape(microwaveRes.values[302000:310000],(-1,MAX_LENGTH))
meanmicrowaveVal = np.nan_to_num(microwaveVal).mean(axis=1)
microwaveVal-=meanmicrowaveVal.reshape((meanmicrowaveVal.shape[0]), 1)

ovenVal=np.reshape(ovenRes.values[302000:310000],(-1,MAX_LENGTH))
meanovenVal = np.nan_to_num(ovenVal).mean(axis=1)
ovenVal-=meanovenVal.reshape((meanovenVal.shape[0]), 1)

fridgeVal=np.reshape(fridgeRes.values[302000:310000],(-1,MAX_LENGTH))
meanfridgeVal = np.nan_to_num(fridgeVal).mean(axis=1)
fridgeVal-=meanXVal.reshape((meanfridgeVal.shape[0]), 1)

# remove the test sets with missing values
maskDropVal = np.ones(len(XVal), dtype=bool)
maskDropVal[np.where(np.any(np.isnan(XVal),axis=1))[0]] = False
XVal=XVal[maskDropVal,...]
fridgeVal=fridgeVal[maskDropVal,...]
ovenVal=ovenVal[maskDropVal,...]
microwaveVal=microwaveVal[maskDropVal,...]
maskVal=maskVal[maskDropVal,...]
print (np.shape(XVal))
# convert to theano float types
X=X.astype(theano.config.floatX)
mask=mask.astype(theano.config.floatX)
fridgeY=fridgeY.astype(theano.config.floatX)

XVal=XVal.astype(theano.config.floatX)
maskVal=maskVal.astype(theano.config.floatX)
fridgeVal=fridgeVal.astype(theano.config.floatX)

# Min/max sequence length

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 1200
# Number of training sequences in each batch
N_BATCH = 252
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 10
# Number of epochs to train the net
NUM_EPOCHS = 10
raise SystemExit
