# Import Library
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


# Find peaks
def peak_detect(price,order):
    
    max_idx = list(argrelextrema(price,np.greater,order=order)[0])
    mix_idx = list(argrelextrema(price,np.less,order=order)[0])
    
    idx = max_idx + mix_idx + [len(price)-1]
    
    idx.sort()
    
    current_idx = idx[-5:]

    start = min(current_idx)
    end = max(current_idx)

    current_pat = price[current_idx]
    
    return current_idx,current_pat,start,end


# Gartley function pattern
def is_gartley(moves,error_allowed):
    
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    
    AB_range = np.array([0.618 - error_allowed, 0.618 + error_allowed])*abs(XA)
    BC_range = np.array([0.382 - error_allowed, 0.886 + error_allowed])*abs(AB)
    CD_range = np.array([1.27 - error_allowed, 1.618 + error_allowed])*abs(BC)
    
    # Bullish pattern
    if XA>0 and AB<0 and BC>0 and CD<0:
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
    
            return 1
            
        else:
            
            return np.NAN  
            
    # Bearish pattern        
    elif XA<0 and AB>0 and BC<0 and CD>0:
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
           
            return -1
            
        else:
            
            return np.NAN  

    else:
        
        return np.NAN  
    
# Butterly function pattern
def is_butterfly(moves,error_allowed):
    
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    
    AB_range = np.array([0.786 - error_allowed, 0.786 + error_allowed])*abs(XA)
    BC_range = np.array([0.382 - error_allowed, 0.886 + error_allowed])*abs(AB)
    CD_range = np.array([1.618- error_allowed, 2.618 + error_allowed])*abs(BC)
    
    # Bullish pattern
    if XA>0 and AB<0 and BC>0 and CD<0:
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
    
            return 1
            
        else:
            
            return np.NAN  
            
    # Bearish pattern        
    elif XA<0 and AB>0 and BC<0 and CD>0:
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
           
            return -1
            
        else:
            
            return np.NAN  

    else:
        
        return np.NAN   
    
# Bat function pattern
def is_bat(moves,error_allowed):
    
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    
    AB_range = np.array([0.382 - error_allowed, 0.5 + error_allowed])*abs(XA)
    BC_range = np.array([0.382 - error_allowed, 0.886 + error_allowed])*abs(AB)
    CD_range = np.array([1.618- error_allowed, 2.618 + error_allowed])*abs(BC)
    
    # Bullish pattern
    if XA>0 and AB<0 and BC>0 and CD<0:
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
    
            return 1
            
        else:
            
            return np.NAN  
            
    # Bearish pattern        
    elif XA<0 and AB>0 and BC<0 and CD>0:
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
           
            return -1
            
        else:
            
            return np.NAN  

    else:
        
        return np.NAN    
    
    
# Crab function pattern
def is_crab(moves,error_allowed):
    
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    
    AB_range = np.array([0.382 - error_allowed, 0.618 + error_allowed])*abs(XA)
    BC_range = np.array([0.382 - error_allowed, 0.886 + error_allowed])*abs(AB)
    CD_range = np.array([2.24- error_allowed, 3.618 + error_allowed])*abs(BC)
    
    # Bullish pattern
    if XA>0 and AB<0 and BC>0 and CD<0:
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
    
            return 1
            
        else:
            
            return np.NAN  
            
    # Bearish pattern        
    elif XA<0 and AB>0 and BC<0 and CD>0:
    
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
           
            return -1
            
        else:
            
            return np.NAN  

    else:
        
        return np.NAN  
 
# Trailing stop 
def walk_forward(price,sign,slippage,stop):
    
    slippage = float(slippage)/float(10000)
    stop_amount = float(stop)/float(10000)        
    
    if sign == 1:
        
        initial_stop_loss = price[0] - stop_amount
        
        stop_loss = initial_stop_loss
        
        for i in range(1,len(price)):
            
            move = price[i] - price [i-1]
            
            if move > 0 and (price[i] - stop_amount) > initial_stop_loss:
                
                stop_loss = price[i] - stop_amount
    
            elif price[i] < stop_loss:
                
                return stop_loss - price[0] - slippage
    
    elif sign == -1:
        
        initial_stop_loss = price[0] + stop_amount
        
        stop_loss = initial_stop_loss
        
        for i in range(1,len(price)):
            
            move = price[i] - price [i-1]
            
            if move < 0 and (price[i] + stop_amount) < initial_stop_loss:
                
                stop_loss = price[i] + stop_amount
    
            elif price[i] > stop_loss:
                
                return price[0] - stop_loss - slippage    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    