from feature_functions import *
import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objs as go
from plotly import tools


# (1) Load CSV & create moving average

df = pd.read_csv('F:\Marc\FX\Data\EURUSD_Hourly.csv')
df.columns = ['date','open','high','low','close','volume']
df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open','high','low','close','volume']]
df = df.drop_duplicates(keep=False)
df = df.iloc[:200]

ma = df.close.rolling(center=False,window=30).mean()


# (2) Get function data from selected function

HAresults = heikenashi(df,[1])
HA = HAresults.candles[1]


# (3) Plot

trace0 = go.Ohlc(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,name='Currency Quote')
trace1 = go.Scatter(x=df.index,y=ma)
trace2 = go.Ohlc(x=HA.index,open=HA.open,high=HA.high,low=HA.low,close=HA.close,name='Heiken Ashi')
          

data = [trace0,trace1,trace2]

fig = tools.make_subplots(rows=2,cols=1,shared_xaxes=True,shared_yaxes=True)
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)

py.offline.plot(fig,filename='tutorial_3.html')