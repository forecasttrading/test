import pandas as pd
import plotly as py
from plotly import tools
import plotly.graph_objs as go


df = pd.read_csv('F:\Marc\FX\Data\EURUSD_Hourly.csv')
df.columns = ['date','open','high','low','close','volume']
df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open','high','low','close','volume']]
df = df.drop_duplicates(keep=False)

trace = go.Ohlc(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,name='Currency Quote')
                
data = [trace]

py.offline.plot(data,filename='tutorial.html')