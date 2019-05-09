# Import Library
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

#----------------------------------------------------------------------------------------------------------

# Load CSV data

#data = pd.read_csv('F:\Marc\FX\Project\Test_1\Test 1\Pattern Scanning\Data\EURUSD_Hourly.csv')
#data.columns = ['date','open','high','low','close','vol']
#data.date = pd.to_datetime(data.date,format='%d.%m.%Y %H:%M:%S.%f')
#data = data.set_index(data.date)
#data = data[['open','high','low','close','vol']]
#data = data.drop_duplicates(keep=False)

#price = data.close.iloc[:500]
#price = data.close.copy()

#----------------------------------------------------------------------------------------------------------


style.use('ggplot')

# Save CSV data from Yahoo
#start = dt.datetime(2000,1,1)
#end = dt.datetime(2018,12,31)
#df = web.DataReader('TSLA','yahoo',start,end)
#df.to_csv('tsla.csv')


# Read CSV data
df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
#df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)

df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)







ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g', colordown='r')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

plt.show()











