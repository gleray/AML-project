import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
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

#trainning=0:10000
plt.plot(dataRes[:8640],"b",fridgeRes[:8640],"r",microwaveRes[:8640],"g",dishwasher[:8640],"k")
plt.show()
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

