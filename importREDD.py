import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
os.chdir('/run/user/1000/gvfs/dav:host=data.deic.dk,ssl=true,user=gleray%40elektro.dtu.dk,prefix=%2Fremote.php%2Fwebdav/Data/energy_disaggregation')


data=pd.read_csv('REDD/low_frequency/house_1/channel_1.dat',header=None,sep=" ")

def dataAggregate(data,step=4):
	return(data.reshape(-1,step).sum(axis=1))

def missingElements(L):
	start, end = L[0], L[len(L)-1]
	LFull=pd.Series(range(start, end + 1))
	LFullBoolean=~LFull.isin(L)
	LFullBinary=LFullBoolean.astype(int)
	dLFullBinary=LFullBinary.diff(1)
	noNandLfullBin=np.nan_to_num(dLFullBinary)
	beginMiss=LFull[np.where(noNandLfullBin==1.)[0]].values
	endMiss=LFull[np.where(noNandLfullBin==-1.)[0]].values
	return(beginMiss,endMiss)
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
print begin[0]
print end[0]
beginAdj=[x-1 for x in begin]
endAdj=end
print np.where(data[0]==beginAdj[0])[0]
print data.ix[np.where(data[0]==beginAdj[0])[0][0]:np.where(data[0]==endAdj[0])[0][0]]


# Mytest=pd.Series([0,0,0,0,1,1,1,1,0,0,0])
