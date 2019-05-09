# Import Library
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


# Load CSV data

data = pd.read_csv('F:\Marc\FX\Project\Test_1\Test 1\Pattern Scanning\Data\EURUSD_Hourly.csv')
data.columns = ['date','open','high','low','close','vol']
data.date = pd.to_datetime(data.date,format='%d.%m.%Y %H:%M:%S.%f')
data = data.set_index(data.date)
data = data[['open','high','low','close','vol']]
data = data.drop_duplicates(keep=False)

#price = data.close.iloc[:500]
price = data.close.copy()


# Find relative extrema

error_allowed = 10.0/100

for i in range(100,len(price)):

    max_idx = list(argrelextrema(price.values[:i],np.greater,order=10)[0])
    mix_idx = list(argrelextrema(price.values[:i],np.less,order=10)[0])
    
    idx = max_idx + mix_idx + [len(price.values[:i])-1]
    
    idx.sort()
    
    current_idx = idx[-5:]

    start = min(current_idx)
    end = max(current_idx)

    current_pat = price.values[current_idx]
    
    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]
    
    # Bullish pattern
    if XA>0 and AB<0 and BC>0 and CD<0:
        
        AB_range = np.array([0.618 - error_allowed, 0.618 + error_allowed])*abs(XA)
        BC_range = np.array([0.382 - error_allowed, 0.886 + error_allowed])*abs(AB)
        CD_range = np.array([1.27 - error_allowed, 1.618 + error_allowed])*abs(BC)
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
        
            plt.plot(np.arange(start,i+15),price.values[start:i+15])
            plt.plot(current_idx,current_pat,c='r')
            plt.show()
            
    # Bearish pattern        
    elif XA<0 and AB>0 and BC<0 and CD>0:
        
        AB_range = np.array([0.618 - error_allowed, 0.618 + error_allowed])*abs(XA)
        BC_range = np.array([0.382 - error_allowed, 0.886 + error_allowed])*abs(AB)
        CD_range = np.array([1.27 - error_allowed, 1.618 + error_allowed])*abs(BC)
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
        
            plt.plot(np.arange(start,i+15),price.values[start:i+15])
            plt.plot(current_idx,current_pat,c='r')
            plt.show()









