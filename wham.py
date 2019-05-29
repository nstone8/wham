import os
import pandas as pd
import numpy as np
import math
import io
from matplotlib import pyplot as plt
import scipy.special

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
    cmdFile=open(os.path.join(savePath,'command.txt'),'w')
    cmdFile.write(whamCommand)
    cmdFile.close()
    print(whamCommand)

def plotResults(resultPath:str,style='point'):
    resultsFrame=loadResults(resultPath)
    fmtString=''
    if style=='point':
        fmtString='.'
    elif style=='line':
        fmtString='-'
    plt.plot(resultsFrame.iloc[:,0],resultsFrame.iloc[:,1],fmtString)
    plt.show()

def loadResults(resPath:str)->pd.DataFrame:
    buf=io.StringIO()
    resultFile=open(resPath)
    hashCount=0
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
    return resultsFrame

def multiSigmoid(zpos,*args)->'list':
    '''returns the sum of multiple sigmoids for values of zpos
    Parameters:
    zpos: series of values to calculate the sum of sigmoids over
    args: a series of n depth values, n widths and n centers, where n is the number of sigmoids'''
    if len(args)%3 != 0:
        raise Exception('The number of arguments must be a multiple of 3')
    nSigmoids=int(len(args)/3)
    depths=args[:nSigmoids]
    widths=args[nSigmoids:2*nSigmoids]
    centers=args[2*nSigmoids:]
    energies=[]
    for z in zpos:
        thisEnergy=0
        for d,w,c in zip(depths,widths,centers):
            thisEnergy+=-1*d*(scipy.special.expit((1/w)*(z-c))-1)
        energies.append(thisEnergy)
    return energies

def multiTanh(zpos,*args)->'list':
    '''returns the sum of multiple hyperbolic tangent sigmoids for values of zpos
    Parameters:
    zpos: series of values to calculate the sum of sigmoids over
    args: a series of n depth values, n widths and n centers, where n is the number of sigmoids'''
    if len(args)%3 != 0:
        raise Exception('The number of arguments must be a multiple of 3')
    nSigmoids=int(len(args)/3)
    depths=args[:nSigmoids]
    widths=args[nSigmoids:2*nSigmoids]
    centers=args[2*nSigmoids:]
    energies=[]
    for z in zpos:
        thisEnergy=0
        for d,w,c in zip(depths,widths,centers):
            thisEnergy+=-0.5*d*(np.tanh((1/w)*(z-c))-1)
        energies.append(thisEnergy)
    return energies


def fitLandscape(landscape:str,smooth:int=5):
    data=loadResults(landscape)
    zpos=data.iloc[:,0]
    en=data.iloc[:,1]
    enSmoothed=[]
    dSmooth=int((smooth-1)/2)
    zposSmoothed=[]
    for j in range(dSmooth,len(en)-dSmooth-1):
        enSmoothed.append(np.mean(en[j-dSmooth:j+dSmooth+1]))
        zposSmoothed.append(zpos[j])
    ddEn=[] #second derivative of energy
    ddZpos=[]
    for i in range(1,len(enSmoothed)-1): #calculate second derivative of energy-ish
        ddEn.append(enSmoothed[i+1]-(2*enSmoothed[i])+enSmoothed[i-1])
        ddZpos.append(zposSmoothed[i])
    fig,axes=plt.subplots(nrows=1,ncols=3,sharex=True)
    axes[0].plot(zpos,en)
    axes[1].plot(zposSmoothed,enSmoothed)
    axes[2].plot(ddZpos,ddEn)
    plt.show()
