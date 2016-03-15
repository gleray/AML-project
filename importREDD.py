from __future__ import print_function
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import theano
import theano.tensor as T
import lasagne

os.chdir('/run/user/1000/gvfs/dav:host=data.deic.dk,ssl=true,user=gleray%40elektro.dtu.dk,prefix=%2Fremote.php%2Fwebdav/Data/energy_disaggregation')


data=pd.read_csv('REDD/low_frequency/house_1/channel_1.dat',header=None,sep=" ")
fridge=pd.read_csv('REDD/low_frequency/house_1/channel_5.dat',header=None,sep=" ")
microwave=pd.read_csv('REDD/low_frequency/house_1/channel_11.dat',header=None,sep=" ")
dishwasher=pd.read_csv('REDD/low_frequency/house_1/channel_6.dat',header=None,sep=" ")


data[0] = pd.to_datetime(data[0], unit='s')
data = data.set_index(0).rename(columns={1: 'House1'})
dataRes=data.resample('10s',how=sum,limit=5,loffset="10s")
#print dataRes.ix[100000:100005]

fridge[0] = pd.to_datetime(fridge[0], unit='s')
fridge = fridge.set_index(0).rename(columns={1: 'Fridge'})
fridgeRes=fridge.resample('10s',how=sum,limit=5)
#print fridgeRes.ix[100000:100005]

microwave[0] = pd.to_datetime(microwave[0], unit='s')
microwave = microwave.set_index(0).rename(columns={1: 'Microwave'})
microwaveRes=microwave.resample('10s',how=sum,limit=5)
#print microwaveRes.ix[100000:100005]

dishwasher[0] = pd.to_datetime(dishwasher[0], unit='s')
dishwasher = dishwasher.set_index(0).rename(columns={1: 'dishwasher'})
dishwasherRes=dishwasher.resample('10s',how=sum,limit=5)
#print dishwasherRes.ix[100000:100005]

maskMat=dataRes.notnull().astype(int)
mask=np.reshape(maskMat.values[:302400],(-1,8640))
dataRes=np.reshape(dataRes.values[:302400],(-1,8640))
dataRes-=np.nan_to_num(dataRes).reshape(-1,8640).mean(axis=0)
dishwasherRes=np.reshape(dishwasherRes.values[:302400],(-1,8640))
dishwasherRes-=np.nan_to_num(dishwasherRes).reshape(-1,8640).mean(axis=0)
microwaveRes=np.reshape(microwaveRes.values[:302400],(-1,8640))
microwaveRes-=np.nan_to_num(microwaveRes).reshape(-1,8640).mean(axis=0)
fridgeRes=np.reshape(fridgeRes.values[:302400],(-1,8640))
fridgeRes-=np.nan_to_num(fridgeRes).reshape(-1,8640).mean(axis=0)

Y=np.dstack((microwaveRes,dishwasherRes,fridgeRes))
X=dataRes
X_val=
# Min/max sequence length
MAX_LENGTH = 8640
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 35
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10

raise SystemExit
def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
    # The network also needs a way to provide a mask for each sequence.  We'll
    # use a separate input layer for that.  Since the mask only determines
    # which indices are part of the sequence for each batch entry, they are
    # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))

    # We're using a bidirectional network, which means we will combine two
    # RecurrentLayers, one with the backwards=True keyword argument.
    # Setting a value for grad_clipping will clip the gradients in the layer
    # Setting only_return_final=True makes the layers only return their output
    # for the final time step, which is all we need for this task
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)#, only_return_final=True
   
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
         backwards=True)#only_return_final=True,

    # Now, we'll concatenate the outputs to combine them.
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])

    # Our output layer is a simple dense connection, with 1 output unit
    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=3, nonlinearity=lasagne.nonlinearities.tanh)

    target_values = T.vector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    # The network output will have shape (n_batch, 1); let's flatten to get a
    # 1-dimensional vector of predicted values
    predicted_values = network_output.flatten()

    # Our cost will be mean-squared error
    cost = T.mean((predicted_values - target_values)**2)

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute stochastic gradient descent updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = gen_data()

    print("Training ...")
    try:
        for epoch in range(num_epochs):
            for _ in range(EPOCH_SIZE):
                X, y, m = gen_data()
                train(X, y, m)
            cost_val = compute_cost(X_val, y_val, mask_val)
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
    except KeyboardInterrupt:
        pass
#plt.plot(dataRes[302400:],"b")
#plt.show()
#raise SystemExit
# todo, check that datatimestep<step
# def dataAggregate(data,step=4):
# 	#takes the data and a step (4 by default)
# 	#return the data with the time step corresponding
# 	return(data.reshape(-1,step).sum(axis=1))

# # todo generalise to other timestep than 1s
# def missingElements(L):
# 	#takes L the actual timestamp 
# 	#return to location of the missing elements
# 	start, end = L[0], L[len(L)-1]
# 	LFull=pd.Series(range(start, end + 1))
# 	LFullBoolean=~LFull.isin(L)
# 	LFullBinary=LFullBoolean.astype(int)
# 	dLFullBinary=LFullBinary.diff(1)
# 	noNandLfullBin=np.nan_to_num(dLFullBinary)
# 	beginMiss=LFull[np.where(noNandLfullBin==1.)[0]].values
# 	endMiss=LFull[np.where(noNandLfullBin==-1.)[0]].values
# 	return(beginMiss,endMiss)

# def fillMissingElements(data,beginM,endM):
# 	#takes data the original dataset and beginM, endM location of the missing data
# 	# fill them with the avg of beginM-1 and endM+1
# 	beginAdj=[x-1 for x in beginM]
# 	endAdj=endM
# 	dataToAdd=pd.DataFrame([[0,0]],columns=list('01'))
# 	# lengthMissChunk=endM-beginM
# 	# missToFillLoc=np.where(lengthMissChunk<=4)
# 	# frames=[pd.DataFrame({'0':np.arange((beginAdj[x]+1),(endAdj[x]),1),
# 	#'1':np.repeat(np.mean(data[1].loc[np.where(data[0]==beginAdj[x])[0][0]:np.where(data[0]==endAdj[x])[0][0]]),endAdj[x]-beginAdj[x]-1)}) for x in missToFillLoc]
# 	#dataToAdd=pd.concat(frames)
# 	for missChunk in range(len(endM)-1):
# 		if endM[missChunk]-beginM[missChunk]-1<=4:
# 			dataToAdd=dataToAdd.append(pd.DataFrame({'0':np.arange((beginAdj[missChunk]+1),(endAdj[missChunk]),1),
# 			'1':np.repeat(np.mean(data[1].loc[np.where(data[0]==beginAdj[missChunk])[0][0]:np.where(data[0]==endAdj[missChunk])[0][0]]),endAdj[missChunk]-beginAdj[missChunk]-1)}), ignore_index=True)
# 	return(dataToAdd)

