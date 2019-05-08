# Import Library
from feature_functions import *
import pandas as pd
import numpy as np


# Load CSV data

#data = pd.read_csv('F:\Marc\FX\Data\EURUSD_Hourly.csv')
#data.columns = [['date','open','high','low','close','AskVol']]
#data = data.set_index(pd.to_datetime(data.date))
#data = data[['open','high','low','close','AskVol']]
#prices = data.drop_duplicates(keep=False)

data = pd.read_csv('F:\Marc\FX\Data\EURUSD_Hourly.csv')
data.columns = ['date','open','high','low','close','AskVol']
data.date = pd.to_datetime(data.date,format='%d.%m.%Y %H:%M:%S.%f')
data = data.set_index(data.date)
data = data[['open','high','low','close','AskVol']]
prices = data.drop_duplicates(keep=False)


# Create listes for each period required by our functions

momentumKey = [3,4,5,8,9,10]
stochasticKey = [3,4,5,8,9,10]
williamsKey = [6,7,8,9,10]
procKey = [12,13,14,15]
wadlKey = [15]
adoscKey = [2,3,4,5]
macdKey = [15,30]
cciKey = [15]
bollingerKey = [15]
heikenashiKey = [15]
paverageKey = [2]
slopeKey = [3,4,5,10,20,30]
fourierKey = [10,20,30]
sineKey = [5,6]

keyList = [momentumKey,stochasticKey,williamsKey,procKey,wadlKey,adoscKey,macdKey,cciKey,bollingerKey,
           heikenashiKey,paverageKey,slopeKey,fourierKey,sineKey]

# Calculate all of the features

momentumDict = momentum(prices,momentumKey)
print('1')

stochasticDict = stochastic(prices,stochasticKey)
print('2')

williamsDict = williams(prices,williamsKey)
print('3')

procDict = proc(prices,procKey)
print('4')

wadlDict = wadl(prices,wadlKey)
print('5')

adoscDict = adosc(prices,adoscKey)
print('6')

macdDict = macd(prices,macdKey)
print('7')

cciDict = cci(prices,cciKey)
print('8')

bollingerDict = bollinger(prices,bollingerKey,2)
print('9')

hkaprices = prices.copy()
hkaprices['Symbol'] = 'SYMB'
HKA = OHLCresample(hkaprices,'15H')
heikenashiDict = heikenashi(HKA,heikenashiKey)
print('10')

paverageDict = paverage(prices,paverageKey)
print('11')

slopeDict = slopes(prices,slopeKey)
print('12')

fourierDict = fourier(prices,fourierKey)
print('13')

sineDict = sine(prices,sineKey)
print('14')


# Create list of dictionaries

dictList = [momentumDict.close,stochasticDict.close,williamsDict.close,procDict.proc,wadlDict.wadl,
            adoscDict.AD,macdDict.line,cciDict.cci,bollingerDict.bands,heikenashiDict.candles,paverageDict.avs,
            slopeDict.slope,fourierDict.coeffs,sineDict.coeffs]


# List of 'base' columns names

colFeat = ['momentum','stoch','will','proc','wadl','adosc','macd','cci','bollinger','heiken','paverage',
           'slope','fourier','sine']


# Populate the masterframe

masterFrame = pd.DataFrame(index=prices.index)
 
for i in range(0,len(dictList)):
    
    if colFeat[i] == 'macd':
        
        #colID = colFeat[i] + str(keyList[6][0]) + str(keyList[6][0])
        colID = colFeat[i] + str(keyList[6][0]) + str(keyList[6][1])
        
        masterFrame[colID] = dictList[i]

    else:
         
        for j in keyList[i]:
            
            for k in list(dictList[i][j]):
                
                colID = colFeat[i] + str(j) + k[0]

                masterFrame[colID] = dictList[i][j][k]

threshold = round(0.7*len(masterFrame))

masterFrame[['open','high','low','close']] = prices[['open','high','low','close']]


# Heiken Ashi is resampled => empty data in between

masterFrame.heiken15open = masterFrame.heiken15open.fillna(method='bfill')
masterFrame.heiken15high = masterFrame.heiken15high.fillna(method='bfill')
masterFrame.heiken15low = masterFrame.heiken15low.fillna(method='bfill')
masterFrame.heiken15close = masterFrame.heiken15close.fillna(method='bfill')


# Drop columns that have 30% or more NAN data

masterFrameCleaned = masterFrame.copy()

masterFrameCleaned = masterFrameCleaned.dropna(axis=1,thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)

masterFrameCleaned.to_csv('F:\Marc\FX\Data\EURUSD_masterFrame.csv')

print('Compled Feature Calculations')



