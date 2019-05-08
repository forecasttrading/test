import pandas as pd


df = pd.read_csv('F:\Marc\FX\Data\EURUSD_Hourly.csv')
df.columns = ['date','open','high','low','close','volume']

df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)

df = df[['open','high','low','close','volume']]

df = df.drop_duplicates(keep=False)

print(df.head())

