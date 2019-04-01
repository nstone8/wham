import os
import pandas as pd
import numpy as np
import math
import io
from matplotlib import pyplot as plt

def createFiles(dataPath:str,savePath:str,winSize:int,temp:float,k:float,bin_width:float=.05,tol=1E-12):
    """dataPath: Path to data .csv file
    savePath: Path to new directory to store output
    winSize: number of points in each umbrella
    bin_width: width of bins in nm
    temp: Reaction temperature
    k: spring constant in N/m"""
    #k should have units of kJ/(mol*nm^2)
    k=k*(1E-21)*(6.0221409E23)
    if os.path.exists(savePath):
        raise Exception('{} already exists, please choose another save location'.format(savePath))
    os.mkdir(savePath)
    data=pd.read_csv(dataPath)
    #convert data to nm
    data.loc[:,'zSensr']=data.loc[:,'zSensr']*1E9
    data.loc[:,'defl']=data.loc[:,'defl']*1E9
    #calculate tip sample distance
    data.loc[:,'ind']=data.loc[:,'zSensr']-data.loc[:,'defl']
    hist_min,hist_max=np.min(data.ind),-1*float('inf')
    metadataFileName=os.path.join(savePath,'metadata.txt')
    metadata=open(metadataFileName,'w')
    tsDir=os.path.join(savePath,'timeseries')
    os.mkdir(tsDir)
    umbrellaIndex=0
    while data.shape[0] >= winSize:
        print('data.shape: ',data.shape)
        #get data for this umbrella
        thisSegment=data.iloc[:(winSize-1),:]
        #delete data from this umbrella from source dataframe
        data=data.iloc[winSize:,:]
        #make a file for this timeseries
        thisSegmentFileName=os.path.join(tsDir,'umbrella{}.txt'.format(umbrellaIndex))
        umbrellaIndex+=1
        thisSegmentFile=open(thisSegmentFileName,'w')
        #loc_win_min should be the mean of the zPos for this window
        loc_win_min=np.mean(thisSegment.zSensr)
        metadata.write('{0}\t{1}\t{2}\n'.format(thisSegmentFileName,loc_win_min,k))
        #format data such that each line has the 'time' (data point number) and ind separated by a tab
        formattedSegmentData=('{0}\t{1}\n'.format(line[0],line.ind) for linenum,line in thisSegment.iterrows())
        thisSegmentFile.writelines(formattedSegmentData)
        thisSegmentFile.close()
        this_hist_max=np.max(thisSegment.ind)
        if this_hist_max>hist_max:
            hist_max=this_hist_max
    metadata.close()
    whamCommand='wham {hist_min} {hist_max} {num_bins} {tol} {temperature} {numpad} {metadatafile} {freefile}'.format(hist_min=hist_min,hist_max=hist_max,num_bins=math.ceil((hist_max-hist_min)/bin_width),tol=tol,temperature=temp,numpad=0,metadatafile=metadataFileName,freefile=os.path.join(savePath,'results.txt'))
    print(whamCommand)
        
def plotResults(resultPath:str,style='point'):
    buf=io.StringIO()
    resultFile=open(resultPath)
    hashCount=0
    fmtString=''
    if style=='point':
        fmtString='.'
    elif style=='line':
        fmtString='-'
    while True:
        thisLine=resultFile.readline()
        if thisLine[0]=='#':
            hashCount+=1
        if hashCount>1:
            break
        buf.write(thisLine)
    resultFile.close()
    buf.seek(0)
    resultsFrame=pd.read_csv(buf,sep='\t')
    plt.plot(resultsFrame.iloc[:,0],resultsFrame.iloc[:,1],fmtString)
    plt.show()
