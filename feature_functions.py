# Import Library
import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from matplotlib.finance import _candlestick
#from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num
from datetime import datetime


class holder:
    1

    
# Heiken Ashi Candles
    
def heikenashi(prices,periods):
    
    """
    /param/ prices: dataframe of OHLC & volume data
    /param/ periods: periods for which to create the candles
    /return/ heiken ashi OHLC candles
    
    """
    
    results = holder()
    dict = {}
    
    HAclose = prices[['open','high','close','low']].sum(axis=1)/4
    
    HAopen = HAclose.copy()
    
    HAopen.iloc[0] = HAclose.iloc[0]
    
    HAhigh = HAclose.copy()
    
    HAlow = HAclose.copy()

    for i in range(1,len(prices)):

        HAopen.iloc[i] = (HAopen.iloc[i-1] + HAclose.iloc[i-1])/2
        
        HAhigh.iloc[i] = np.array([prices.high.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).max()
        
        HAlow.iloc[i] = np.array([prices.low.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).min()

    df = pd.concat((HAopen,HAhigh,HAlow,HAclose),axis=1)
    df.columns = [['open','high','close','low']]
        
    df.index = df.index.droplevel(0)
    
    dict[periods[0]] = df
    
    results.candles = dict
    
    return results
    

# Detrender
    
def detrend(prices,method='difference'):
    
    """
    /param/ prices: dataframe of OHLC currency data
    /param/ periods: method by which to dtrend 'linear' or 'difference'
    /return/ the detrended price series
    
    """
    
    if method == 'difference':
        
        detrended = prices.close[1:]-prices.close[:-1].values
        
    elif method == 'linear':
        
        x = np.arange(0,len(prices))
        y = prices.close.values
        
        model = LinearRegression()
        
        model.fit(x.reshape(-1,1),y.reshape(-1,1))
    
        trend = model.predict(x.reshape(-1,1))
    
        trend = trend.reshape((len(prices),))
    
        detrended = prices.close - trend
    
    else:
    
        print('You did not input a valid method for detrending. Choose linear or difference')
    
    return detrended
    
    
# Fourier Series Expension Fitting Function
    
def fseries(x,a0,a1,b1,w):
    
    """
    /param/ x: the hours (independant variable)
    /param/ a0: first Fourier series coefficient
    /param/ a1: second Fourier series coefficient
    /param/ b1: third Fourier series coefficient
    /param/ w:  Fourier series frequency
    /return/ the value of the Fourier function
    
    """  
    
    f = a0 + a1*np.cos(w*x) + b1*np.sin(w*x)
    
    return f
    
    
# Sine Series Expension Fitting Function
    
def sseries(x,a0,b1,w):
    
    """
    /param/ x: the hours (independant variable)
    /param/ a0: first Fourier series coefficient
    /param/ b1: third Fourier series coefficient
    /param/ w:  Fourier series frequency
    /return/ the value of the Fourier function
    
    """  
    
    f = a0 + b1*np.sin(w*x)
    
    return f 
    
    
# Fourier Series Expension Fitting Function
    
def fourier(prices,periods,method='difference'):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: list of periods for which to compute coefficients [3,5,10,...]
    /param/ method: method by which to detrend the data
    /return/ dict of dataframes containing coefficients for said periods
    
    """  
    
    results = holder()
    dict = {}  
    
    # Option to plot the expansion fit for each iteration
    
    plot = False
    
    # Compute the coefficients of the series
    
    detrended = detrend(prices,method)
    
    for i in range(0,len(periods)):
        
        coeffs = []
        
        for j in range(periods[i],len(prices)-periods[i]):
        
            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j]
            
            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)
        
                try:
                    
                    res = scipy.optimize.curve_fit(fseries,x,y)
        
                except (RuntimeError,OptimizeWarning):
                    
                    res = np.empty((1,4))
                    res[0,:] = np.NAN        
        
            if plot == True:
                
                xt = np.linspace(0,periods[i],100)
                yt = fseries(xt,res[0][0],res[0][1],res[0][2],res[0][3])
        
                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                
                plt.show()
        
            coeffs = np.append(coeffs,res[0],axis=0)
        
        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        
        #coeffs = np.array(coeffs).reshape(((len(coeffs)/4,4)))
        coeffs = np.array(coeffs).reshape(((len(coeffs)//4,4)))
        
        df = pd.DataFrame(coeffs,index=prices.iloc[periods[i]:-periods[i]].index)
        
        df.columns = [['a0','a1','b1','w']]
    
        df = df.fillna(method='bfill')
    
        dict[periods[i]] = df
        
    results.coeffs = dict
    
    return results
    
    
# Sine Series Expension Fitting Function
    
def sine(prices,periods,method='difference'):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: list of periods for which to compute coefficients [3,5,10,...]
    /param/ method: method by which to detrend the data
    /return/ dict of dataframes containing coefficients for said periods
    
    """  
    
    results = holder()
    dict = {}  
    
    # Option to plot the expansion fit for each iteration
    
    plot = False
    
    # Compute the coefficients of the series
    
    detrended = detrend(prices,method)
    
    for i in range(0,len(periods)):
        
        coeffs = []
        
        for j in range(periods[i],len(prices)-periods[i]):
        
            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j]
            
            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)
        
                try:
                    
                    res = scipy.optimize.curve_fit(sseries,x,y)
        
                except (RuntimeError,OptimizeWarning):
                    
                    res = np.empty((1,3))
                    res[0,:] = np.NAN        
        
            if plot == True:
                
                xt = np.linspace(0,periods[i],100)
                yt = sseries(xt,res[0][0],res[0][1],res[0][2])
        
                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                
                plt.show()
        
            coeffs = np.append(coeffs,res[0],axis=0)
        
        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        
        #coeffs = np.array(coeffs).reshape(((len(coeffs)/3,3)))
        coeffs = np.array(coeffs).reshape(((len(coeffs)//3,3)))
        
        #df = pd.DataFrame(coeffs,index=prices.iloc[periods[i]:-periods[i]])
        df = pd.DataFrame(coeffs,index=prices.iloc[periods[i]:-periods[i]].index)
        
        df.columns = [['a0','b1','w']]
    
        df = df.fillna(method='bfill')
    
        dict[periods[i]] = df
    
    results.coeffs = dict
    
    return results    
    
    
# Williams Accumulation Distribution Function
    
def wadl(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: list of periods for which to compute coefficients [3,5,10,...]
    /return/ Williams accumulation distribution lines for each period
    
    """  
    
    results = holder()
    dict = {}     
    
    for i in range(0,len(periods)):
        
        WAD = []
    
        for j in range(periods[i],len(prices)-periods[i]):
    
            TRH = np.array([prices.high.iloc[j],prices.close.iloc[j-1]]).max()
            
            TRL = np.array([prices.low.iloc[j],prices.close.iloc[j-1]]).min()
    
            if prices.close.iloc[j] > prices.close.iloc[j-1]:
                
                PM = prices.close.iloc[j] - TRL
    
            elif prices.close.iloc[j] < prices.close.iloc[j-1]:
    
                PM = prices.close.iloc[j] - TRH
    
            elif prices.close.iloc[j] == prices.close.iloc[j-1]:
                
                 PM = 0
    
            else:
                
                print('Unkown errror occured')
    
            AD = PM * prices.AskVol.iloc[j]
    
            WAD = np.append(WAD,AD)
    
        WAD = WAD.cumsum()
        
        WAD = pd.DataFrame(WAD,index=prices.iloc[periods[i]:-periods[i]].index)
    
        WAD.columns = [['close']]
    
        dict[periods[i]] = WAD
    
    results.wadl = dict
    
    return results  
    
        
# Williams Accumulation Distribution Function
    
def OHLCresample(DataFrame,TimeFrame,column='ask'):
    
    """
    /param/ DataFrame: dataframe containing data that we want to resample
    /param/ TimeFrame: timeframe that we want for resampling
    /param/ column: which column we are resampling (bid or ask) default='ask'
    /return/ resampled OHLC data for the given timeframe
    
    """  
    
    grouped = DataFrame.groupby('Symbol')

    if np.any(DataFrame.columns=='Ask'):
        
        if column =='ask':
            ask = grouped['Ask'].resample(TimeFrame).ohlc()
            askVol = grouped['AskVol'].resample(TimeFrame).count()
            resampled = pd.DataFrame(ask)
            resampled['AskVol'] = askVol
    
        elif column =='bid':
            bid = grouped['Bid'].resample(TimeFrame).ohlc()
            bidVol = grouped['BidVol'].resample(TimeFrame).count()
            resampled = pd.DataFrame(bid)
            resampled['BidVol'] = bidVol
      
        else:
            
            raise ValueError('Column must be a string. Either ask or bid')
    
    elif np.any(DataFrame.columns=='close'):

        open = grouped['open'].resample(TimeFrame).ohlc()
        high = grouped['high'].resample(TimeFrame).ohlc()
        low = grouped['low'].resample(TimeFrame).ohlc()
        close = grouped['close'].resample(TimeFrame).ohlc()
        askVol = grouped['AskVol'].resample(TimeFrame).count()
        
        resampled = pd.DataFrame(open)
        resampled['high'] = high
        resampled['low'] = low
        resampled['close'] = close
        resampled['AskVol'] = askVol
            
    resampled = resampled.dropna()
    
    return resampled
    
    
# Momentum Function
    
def momentum(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: list of periods to calculate function value
    /return/ Momentum indicator
    
    """  
    
    results = holder()
    open = {}     
    close = {}  
    
    for i in range(0,len(periods)):
        
        open[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:]-prices.open.iloc[:-periods[i]].values,
                                        index=prices.iloc[periods[i]:].index)
    
        close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:]-prices.close.iloc[:-periods[i]].values,
                                        index=prices.iloc[periods[i]:].index)
        
        open[periods[i]].columns = [['open']]
        close[periods[i]].columns = [['close']]
        
    results.open = open
    results.close = close
    
    return results
    
    
# Stochastic Oscillator Function
    
def stochastic(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: list of periods to calculate function value
    /return/ oscillator function values
    
    """  
    
    results = holder()  
    close = {}  
    
    for i in range(0,len(periods)):
        
        Ks = []
        
        for j in range(periods[i],len(prices)-periods[i]):  
        
            C = prices.close.iloc[j+1]
            H = prices.high.iloc[j-periods[i]:j].max()
            L = prices.low.iloc[j-periods[i]:j].min()
        
            if H == L:
                K = 0
                
            else:
                K = 100*(C-L)/(H-L)
                
            Ks = np.append(Ks,K)
            
        df = pd.DataFrame(Ks,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
        df.columns = [['K']]
        df['D'] = df.K.rolling(3).mean()
        df = df.dropna()
        
        close[periods[i]] = df
        
    results.close = close
        
    return results
        
        
# Williams Oscillator Function
    
def williams(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: list of periods to calculate function value
    /return/ oscillator function values
    
    """  
    
    results = holder()  
    close = {}  
    
    for i in range(0,len(periods)):
        
        Rs = []
        
        for j in range(periods[i],len(prices)-periods[i]):  
        
            C = prices.close.iloc[j+1]
            H = prices.high.iloc[j-periods[i]:j].max()
            L = prices.low.iloc[j-periods[i]:j].min()
        
            if H == L:
                R = 0
                
            else:
                R = -100*(H-C)/(H-L)
                
            Rs = np.append(Rs,R)
            
        df = pd.DataFrame(Rs,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
        df.columns = [['R']]
        df = df.dropna()
        
        close[periods[i]] = df
        
    results.close = close
        
    return results       
        
        
# Price Rate of Change  Function
    
def proc(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: list of periods to calculate function value
    /return/ PROC for indicated periods
    
    """  
    
    results = holder()  
    proc = {}  
    
    for i in range(0,len(periods)):
        
        proc[periods[i]] = pd.DataFrame((prices.close.iloc[periods[i]:]-prices.close.iloc[:-periods[i]].values)\
                                        /prices.close.iloc[:-periods[i]].values)

        proc[periods[i]].columns = [['close']]
        
    results.proc = proc
        
    return results         
        
        
# Accumulation Distribution Oscillator
    
def adosc(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: list of periods to calculate function value
    /return/ indicator value for indicated periods
    
    """  
    
    results = holder()  
    accdist = {}  
    
    for i in range(0,len(periods)):
        
        AD = []
        
        for j in range(periods[i],len(prices)-periods[i]):  
        
            C = prices.close.iloc[j+1]
            H = prices.high.iloc[j-periods[i]:j].max()
            L = prices.low.iloc[j-periods[i]:j].min()
            V = prices.AskVol.iloc[j+1]
        
            if H == L:
                Mult = 0
                
            else:
                Mult = ((C-L)-(H-C))/(H-L)
                
            AD = np.append(AD,Mult*V)
            
        AD = AD.cumsum()
        AD = pd.DataFrame(AD,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
        AD.columns = [['AD']]    

        accdist[periods[i]] = AD
        
    results.AD = accdist
        
    return results         
        

# MACD
    
def macd(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: 1*2 array containing values for the EMA's
    /return/ MACD for the indicated periods
    
    """  
    
    results = holder()  

    EMA1 = prices.close.ewm(span=periods[0]).mean()
    EMA2 = prices.close.ewm(span=periods[1]).mean()

    MACD = pd.DataFrame(EMA1 - EMA2)
    MACD.columns = [['L']]
    
    SigMACD = MACD.rolling(3).mean()
    SigMACD.columns = [['SL']]
    
    results.line = MACD
    results.signal = SigMACD
    
    return results      


# Commodity channel Index
    
def cci(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: period for which to compute the indicator
    /return/ CCI for th indicated periods
    
    """  
    
    results = holder() 
    CCI = {}
    
    for i in range(0,len(periods)):
        
        MA = prices.close.rolling(periods[i]).mean()
        std = prices.close.rolling(periods[i]).std()
    
        D = (prices.close - MA)/std
    
        CCI[periods[i]] = pd.DataFrame((prices.close - MA)/(0.015 * D))
        CCI[periods[i]].columns = [['close']]
    
    results.cci = CCI

    return results      


# Bollinger Bands 
    
def bollinger(prices,periods,deviations):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: period for which to compute the indicator
    /param/ deviations: deviations to use when calculating bands (upper & lower)
    /return/ Bollinger Bands
    
    """  
    
    results = holder() 
    boll = {}
    
    for i in range(0,len(periods)):
        
        mid = prices.close.rolling(periods[i]).mean()
        std = prices.close.rolling(periods[i]).std()

        upper = mid + deviations * std
        lower = mid - deviations * std
    
        df = pd.concat((upper,mid,lower),axis=1)
        df.columns = [['upper','mid','lower']]
    
        boll[periods[i]] = df
    
    results.bands = boll

    return results      


# Price Averages
    
def paverage(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: list of the periods  for which to compute the indicator
    /return/ averages over the given periods
    
    """  
    
    results = holder() 
    avs = {}
    
    for i in range(0,len(periods)):
        
        avs[periods[i]] = pd.DataFrame(prices[['open','high','low','close']].rolling(periods[i]).mean())

    results.avs = avs

    return results      


# Slope Functions
    
def slopes(prices,periods):
    
    """
    /param/ prices: OHLC dataframe
    /param/ periods: periods  for which to compute the indicator
    /return/ slopes over the given periods
    
    """  
    
    results = holder() 
    slope = {}
    
    for i in range(0,len(periods)):
        
        ms = []
        
        for j in range(periods[i],len(prices)-periods[i]):  
        
            y = prices.high.iloc[j-periods[i]:j].values
            x = np.arange(0,len(y))

            res = stats.linregress(x,y=y)
            m = res.slope

            ms = np.append(ms,m)
        
        ms = pd.DataFrame(ms,index=prices.iloc[periods[i]:-periods[i]].index)
        
        ms.columns = [['high']]    

        slope[periods[i]] = ms
        
    results.slope = slope

    return results      


