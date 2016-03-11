import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
os.chdir('/run/user/1000/gvfs/dav:host=data.deic.dk,ssl=true,user=gleray%40elektro.dtu.dk,prefix=%2Fremote.php%2Fwebdav/Data/energy_disaggregation')


data=pd.read_csv('REDD/low_frequency/house_1/channel_1.dat',header=None,sep=" ")

# todo, check that datatimestep<step
def dataAggregate(data,step=4):
	#takes the data and a step (4 by default)
	#return the data with the time step corresponding
	return(data.reshape(-1,step).sum(axis=1))

# todo generalise to other timestep than 1s
def missingElements(L):
	#takes L the actual timestamp 
	#return to location of the missing elements
	start, end = L[0], L[len(L)-1]
	LFull=pd.Series(range(start, end + 1))
	LFullBoolean=~LFull.isin(L)
	LFullBinary=LFullBoolean.astype(int)
	dLFullBinary=LFullBinary.diff(1)
	noNandLfullBin=np.nan_to_num(dLFullBinary)
	beginMiss=LFull[np.where(noNandLfullBin==1.)[0]].values
	endMiss=LFull[np.where(noNandLfullBin==-1.)[0]].values
	return(beginMiss,endMiss)

def fillMissingElements(data,beginM,endM):
	#takes data the original dataset and beginM, endM location of the missing data
	# fill them with the avg of beginM-1 and endM+1
	beginAdj=[x-1 for x in beginM]
	endAdj=endM
	dataToAdd=pd.DataFrame([[0,0]],columns=list('01'))
	# lengthMissChunk=endM-beginM
	# missToFillLoc=np.where(lengthMissChunk<=4)
	# frames=[pd.DataFrame({'0':np.arange((beginAdj[x]+1),(endAdj[x]),1),
	#'1':np.repeat(np.mean(data[1].loc[np.where(data[0]==beginAdj[x])[0][0]:np.where(data[0]==endAdj[x])[0][0]]),endAdj[x]-beginAdj[x]-1)}) for x in missToFillLoc]
	#dataToAdd=pd.concat(frames)
	for missChunk in range(len(endM)-1):
		if endM[missChunk]-beginM[missChunk]-1<=4:
			dataToAdd=dataToAdd.append(pd.DataFrame({'0':np.arange((beginAdj[missChunk]+1),(endAdj[missChunk]),1),
			'1':np.repeat(np.mean(data[1].loc[np.where(data[0]==beginAdj[missChunk])[0][0]:np.where(data[0]==endAdj[missChunk])[0][0]]),endAdj[missChunk]-beginAdj[missChunk]-1)}), ignore_index=True)
	return(dataToAdd)
#print dataAggregate(data[1][:8])

# print data[0][len(data)-1]

# print data[0][1:5]
# print (1306267022-1303132929)
# print(
#     datetime.datetime.fromtimestamp(
#         int("1303984666")
#     ).strftime('%Y-%m-%d %H:%M:%S')
# )
# print(
#     datetime.datetime.fromtimestamp(
#         int("1303984667")
#     ).strftime('%Y-%m-%d %H:%M:%S')
# )
# print(
#     datetime.datetime.fromtimestamp(
#         int("1304133034")
#     ).strftime('%Y-%m-%d %H:%M:%S')
# )
#print len(data)
# plt.plot(np.cumsum(data[0]),"r")
# plt.show()

# for xc in MV:
#     plt.axvline(x=xc)
#plt.plot(data[0],data[1],"r")
#plt.show()
begin,end=missingElements(data[0])
#print begin[0]
#print end[0]
beginAdj=[x-1 for x in begin]
endAdj=end
#print np.where(data[0]==beginAdj[0])[0]
#print data.ix[np.where(data[0]==beginAdj[0])[0][0]:np.where(data[0]==endAdj[0])[0][0]]
val=data[1]
#print data.ix[np.where(data[0]==beginAdj[0])[0][0]:np.where(data[0]==endAdj[0])[0][0]]
#print pd.DataFrame([[0,0]],columns=list("01"))
Mydata=fillMissingElements(data,begin,end)
res=data.append(Mydata)
print res.sort('0',ascending=True)
# Mytest=pd.Series([0,0,0,0,1,1,1,1,0,0,0])