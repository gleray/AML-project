import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import theano.tensor as T
import lasagne
import theano
import scipy.stats as ss
#from IPython import display
store =pd.HDFStore('/run/user/1000/gvfs/dav:host=data.deic.dk,ssl=true,user=gleray%40elektro.dtu.dk,prefix=%2Fremote.php%2Fwebdav/old\ files/Advanced machine learning/project/AML-project/datastore/low_frequency.h5')
appliances=['microwave','refrigerator','stove']
sampleLength=360
sampleSize=50
sampleLength2=sampleLength
numAppliances=len(appliances)
numEpochs=1000
sizeEpochs=25
learningRate=0.005
gradClip=10

finalMask=[list() for _ in xrange(len(store['houses']))]
matches=[list() for _ in xrange(len(store['houses']))]

# generate a mask for each house on where there is enougth data.
for house in xrange(len(store['houses'])):
	#handle labels
	labels=store[store['houses'][house] + '/labels']
	labels=labels.rename(columns={1: 'labels'})
	labels=labels.sort_index(by='labels')
	#matches the labels to the list of appliance to take into the analysis
	matches[house] = [labels.index[j] for j in xrange(len(labels)) if labels.values[j] in appliances]
	#generate a mask of where the data is
	naMaskMain= pd.rolling_sum((store[store['houses'][house] + '/data'][[1,2]]*0+1).fillna(0), window=sampleLength).shift(-sampleLength).sum(axis=1)>=(sampleLength*2-sampleLength*2*0.1)
	naMask= pd.rolling_sum((store[store['houses'][house] + '/data'][matches[house]]*0+1).fillna(0), window=sampleLength).shift(-sampleLength).sum(axis=1)>=(sampleLength*len(matches[house])-sampleLength*len(matches[house])*0.1)
	activationMask = pd.rolling_sum((store[store['houses'][house] + '/data'][matches[house]] > 0), window=sampleLength).shift(-sampleLength).sum(axis=1)>0
	finalMask[house] = activationMask*naMask*naMaskMain




# def RNNNILM(sampleLength,numAppliances,numEpochs,sizeEpochs,learningRate,gradClip):
    
# 	return(Yval,predictions)


def sampleGen(sampleSize,val=False):
	if val:
		house=np.random.randint(0,1)
	else:
		house=np.random.randint(0,len(store['houses']))
	
	labels=store[store['houses'][house] + '/labels']
	labels=labels.rename(columns={1: 'labels'})
	sampleTimes = pd.to_datetime(np.random.choice(finalMask[house][finalMask[house]].index, size=sampleSize), unit='s')

	Y = np.zeros((sampleSize,sampleLength+1,len(appliances)))
	for j in xrange(len(appliances)):
		if appliances[j] in labels.ix[matches[house]].values:

			Y[:,:,j] = np.array([store[store['houses'][house] + '/data'][labels[labels['labels']==appliances[j]].index].ix[s:s+pd.DateOffset(hours=1)].values for s in sampleTimes])[:,:,0]
	if val: 
		probaReg=0
	else:
		probaReg=np.random.random_sample()
	if probaReg<=0.5:
		X = np.array([store[store['houses'][house] + '/data'][[1,2]].ix[s:s+pd.DateOffset(hours=1)].sum(axis=1).values for s in sampleTimes])
		X = np.reshape(X,(np.shape(X)[0],np.shape(X)[1],1))
	else:
		X = np.reshape(np.sum(Y,axis=2),(sampleSize,sampleLength+1,1))
	X=np.log(X+0.00001)
	Y=np.log(Y+0.00001)
	mask=np.nan_to_num(np.array([X[:,:,0]*0+1]))[0,:,:]
	return (mask.astype(theano.config.floatX),X.astype(theano.config.floatX),Y.astype(theano.config.floatX))

def dummySampleGen():
# 	# define 3 appliances microwave, refrigerator, stove.
# 	# randomly pick in a uniform manner where to start the activation period
# 	# microwave 3 levels -> for each activation pick a level | 
# 	# refrigerator 1 level of activation -> Normal distribution to do the activation period and add a bit of gaussian noise
# 	# stove 3 levels of activation-> random int to define how many level of activation 
	activationLevels=[[1.5,2.5,3.5],[1],[2,3,4]]
	mu=[20,70,90]
	sigma=[15,35,60]
	Y = np.zeros((sampleSize,sampleLength2,len(appliances)))
	for j in xrange(len(appliances)):

		for i in xrange(sampleSize):
			activations=np.random.randint(0,sampleLength2,np.random.randint(0,5))

			lengthActivation=np.int_(sigma[j] * np.random.randn(len(activations)) + mu[j]+activations)
			if np.any(lengthActivation>sampleLength2): 
				if len(activations)>1:
					lengthActivation[lengthActivation>sampleLength2]=sampleLength2
				else:
					lengthActivation=sampleLength2
			activationLevel=[activationLevels[j][np.random.randint(0,len(activationLevels[j]))] for activation in xrange(len(activations))]
			if len(activations)!=0:
				if len(activations)>1: 
					for k in xrange(len(activations)): 
						Y[i,range(activations[k],lengthActivation[k]),j]=activationLevel[k]
				else:
					Y[i,range(activations,lengthActivation),j]=activationLevel
	Y = Y+np.random.normal(0,0.005,(sampleSize,sampleLength2,len(appliances)))
	X = np.reshape(np.sum(Y,axis=2),(sampleSize,sampleLength2,1))
	mask=np.nan_to_num(np.array([X[:,:,0]*0+1]))[0,:,:]
	return (mask.astype(theano.config.floatX),X.astype(theano.config.floatX),Y.astype(theano.config.floatX))


raise SystemExit
print("numEpoch = {}, sampleSize = {},learningRate = {}".format(numEpochs, sampleSize,learningRate))

tensorX = T.TensorType('float64', (False,)*numAppliances)
X_sym=tensorX('X')
Y_sym = tensorX('Y')
mask_sym = T.matrix()
l_in = lasagne.layers.InputLayer(shape=(None,sampleLength2,1),input_var=X_sym)

l_mask = lasagne.layers.InputLayer(shape=(None,sampleLength2))

l_forward = lasagne.layers.RecurrentLayer(
            l_in,500,mask_input=l_mask, 
            grad_clipping=gradClip,
            W_in_to_hid=lasagne.init.HeNormal(),
            W_hid_to_hid=lasagne.init.HeNormal(),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
                                                                    #print lasagne.layers.get_output_shape(l_forward)

l_backward = lasagne.layers.RecurrentLayer(
            l_in,500,grad_clipping=gradClip,mask_input=l_mask, 
            W_in_to_hid=lasagne.init.HeNormal(),
            W_hid_to_hid=lasagne.init.HeNormal(),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            backwards=True)

                                                                    # print lasagne.layers.get_output_shape(l_backward)
        
    # Now, we'll concatenate the outputs to combine them.
#l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])
   
                                                                    # print lasagne.layers.get_output_shape(l_concat)
l_reshape_f = lasagne.layers.ReshapeLayer(l_forward, (-1, [2])) 
l_reshape_b = lasagne.layers.ReshapeLayer(l_backward, (-1, [2])) 
l_concat = lasagne.layers.ConcatLayer([l_reshape_f,l_reshape_b])

l_hidden=lasagne.layers.DenseLayer(l_concat, num_units=500, nonlinearity=lasagne.nonlinearities.leaky_rectify)
                                                             #print lasagne.layers.get_output_shape(l_reshape)
    
    # Our output layer is a simple dense connection, with 1 output unit
    #l_hidden = lasagne.layers.DenseLayer(l_forward, num_units=500, nonlinearity=lasagne.nonlinearities.tanh)
l_out = lasagne.layers.DenseLayer(l_hidden, num_units=numAppliances, nonlinearity=lasagne.nonlinearities.linear)
  
                                                                    #print lasagne.layers.get_output_shape(l_out)

predictions = T.reshape(lasagne.layers.get_output(l_out,inputs={l_in: X_sym,l_mask: mask_sym}),(-1,sampleLength2,numAppliances))
loss =  T.mean(lasagne.objectives.squared_error(predictions,Y_sym))

 
# Retrieve all parameters from the network
allParameters = lasagne.layers.get_all_params(l_out, trainable=True)
allGrads = [T.clip(g,-10,10) for g in T.grad(loss, allParameters)]
allGrads = lasagne.updates.total_norm_constraint(allGrads,10)

#print("updates")updates = lasagne.updates.nesterov_momentum(allGrads,allParameters, learning_rate=learningRate,momentum=0.9)
updates = lasagne.updates.adagrad(allGrads,allParameters, learning_rate=learningRate)
#updates = lasagne.updates.sgd(allGrads,allParameters, learning_rate=learningRate)
#print('train_func')
trainFunc = theano.function([X_sym, Y_sym, mask_sym], [loss,predictions], updates=updates)#
#print('compute_cost')
computeCost = theano.function([X_sym, Y_sym, mask_sym], [loss,predictions]) #, mask_sym
#print('start training')
maskval,Xval,Yval=dummySampleGen()
storeRes = pd.HDFStore('datastore/resRNN.h5')
storeRes['validationSet']=pd.Panel(np.transpose(Yval,(2,0,1)), items=appliances,
        		major_axis=xrange(sampleSize),minor_axis=xrange(sampleLength2))
storeRes['validationSetX']=pd.DataFrame(Xval[:,:,0])

storeRes.close()
try:
    for epoch in range(numEpochs):
        for _ in xrange(sizeEpochs):
            mask,X,Y=dummySampleGen()
            cost,pred=trainFunc(np.nan_to_num(X),np.nan_to_num(Y), mask)
            #print(cost)
        
        costVal,predict = computeCost(np.nan_to_num(Xval),np.nan_to_num(Yval),maskval)
        
        storeRes = pd.HDFStore('datastore/resRNN.h5')
        storeRes['Epoch_'+str(epoch)+'/predictions']= pd.Panel(np.transpose(predict,(2,0,1)), items=appliances,
        		major_axis=xrange(sampleSize),minor_axis=xrange(sampleLength2))
        storeRes['Epoch_'+str(epoch)+'/errors']=pd.Series(costVal,index={epoch})
        storeRes.close()
            # plt.plot(sampleVal,errors)
            # plt.ylabel('Validation Error', fontsize=15)
            # plt.xlabel('Processed samples', fontsize=15)
            # plt.title('', fontsize=20)
            # plt.grid('on')
            # display.display(plt.gcf())
            # display.clear_output(wait=True)
            # plt.show()
        print("Epoch {} cost = {}".format(epoch, costVal))
except KeyboardInterrupt:
        pass
        