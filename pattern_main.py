# Import Library
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from harmonic_functions import *
from tqdm import tqdm


# Load CSV data

data = pd.read_csv('F:\Marc\FX\Project\Test_1\Test 1\Pattern Scanning\Data\EURUSD_Hourly.csv')
data.columns = ['date','open','high','low','close','vol']
data.date = pd.to_datetime(data.date,format='%d.%m.%Y %H:%M:%S.%f')
data = data.set_index(data.date)
data = data[['open','high','low','close','vol']]
data = data.drop_duplicates(keep=False)

#price = data.close.iloc[:500]
price = data.close.copy()


# Find peaks

error_allowed = 5.0/100
pnl = []
trade_dates = []
correct_pats = 0
pats = 0

plt.ion()

for i in tqdm(range(100,len(price))):

    current_idx,current_pat,start,end = peak_detect(price.values[:i],order=5)
    
    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]
    
    moves = [XA,AB,BC,CD]
    
    gart = is_gartley(moves,error_allowed)
    butt = is_butterfly(moves,error_allowed)
    bat = is_bat(moves,error_allowed)
    crab = is_crab(moves,error_allowed)

    harmonics = np.array([gart, butt, bat, crab])
    labels = ['Gartley', 'Butterlfy', 'Bat', 'Crab']
        
    if np.any(harmonics == 1) or np.any(harmonics == -1):
        
        pats +=1
        
        for j in range(0,len(harmonics)):
            
            if harmonics[j] == 1 or harmonics[j] == -1:
                
                sense = 'Bearish' if harmonics[j] == -1 else 'Bullish'
                label =  sense + labels[j] + 'Found'  
                
                start = np.array(current_idx).min()
                end = np.array(current_idx).max()
                date = data.iloc[end].name
                trade_dates = np.append(trade_dates,date)

                pips = walk_forward(price.values[end:],harmonics[j],slippage=4,stop=15)
                
                pnl = np.append(pnl,pips)

                cumpips = pnl.cumsum()
                    
                if pips>0:
                    
                    correct_pats +=1

                lbl = 'Accuracy ' + str(100*float(correct_pats)/float(pats)) + ' %'
                
                plt.clf()
                plt.plot(cumpips,label=lbl)
                plt.legend()
                plt.pause(0.05)    
                 
                #plt.title(label)
                #plt.plot(np.arange(start,i+15),price.values[start:i+15])
                #plt.plot(current_idx,current_pat,c='r')
                #plt.show()










