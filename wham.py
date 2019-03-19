import os
import pandas as pd
import numpy as np
def createFiles(dataPath:str,savePath:str,winSize:int,temp:float,k:float,tol=1E-12):
    """dataPath: Path to data .csv file
    savePath: Path to new directory to store output
    winSize: number of points in each umbrella
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
    data.loc[:,'tsd']=data.loc[:,'defl']-data.loc[:,'zSensr']
    hist_min,hist_max=float('inf'),np.max(data.tsd)
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
        loc_win_min=np.mean(thisSegment.tsd)
        metadata.write('{0}\t{1}\t{2}\n'.format(thisSegmentFileName,loc_win_min,k))
        #format data such that each line has the 'time' (data point number) and tsd separated by a tab
        formattedSegmentData=('{0}\t{1}\n'.format(line[0],line.tsd) for linenum,line in thisSegment.iterrows())
        thisSegmentFile.writelines(formattedSegmentData)
        thisSegmentFile.close()
        this_hist_min=np.min(thisSegment.tsd)
        if this_hist_min<hist_min:
            hist_min=this_hist_min
    metadata.close()
    whamCommand='wham {hist_min} {hist_max} {num_bins} {tol} {temperature} {numpad} {metadatafile} {freefile}'.format(hist_min=hist_min,hist_max=hist_max,num_bins=umbrellaIndex,tol=tol,temperature=temp,numpad=0,metadatafile=metadataFileName,freefile=os.path.join(savePath,'results.txt'))
    print(whamCommand)
        
