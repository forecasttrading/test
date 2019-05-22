# Import Library
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from tqdm import tqdm
from ta import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Load CSV data
data = pd.read_csv('F:\Marc\FX\Project\Test_1\Test 1\Technical Indicators\Data\EURUSD_Hourly.csv')
data.columns = ['date','open','high','low','close','vol']
data.date = pd.to_datetime(data.date,format='%d.%m.%Y %H:%M:%S.%f')
data = data.set_index(data.date)
data = data[['open','high','low','close','vol']]
data = data.drop_duplicates(keep=False)


# Create a dataframe with all features
features = data.copy()
features = add_all_ta_features(features, "open", "high", "low", "close", "vol", fillna=True)


# Create a dataframe with outcomes
outcomes = pd.DataFrame(index=data.index)

outcomes['close_1d'] = data.close.pct_change(periods=-1)  # next day's returns
outcomes['close_5d'] = data.close.pct_change(periods=-5)  # next 5 day's returns
outcomes['close_10d'] = data.close.pct_change(periods=-10)  # next 10 day's week's returns
outcomes['close_20d'] = data.close.pct_change(periods=-20)  # next 20 day's  returns


# Create x and y series
y = outcomes.close_1d
x = features

xy = x.join(y).dropna()

y = xy[y.name]
x = xy[x.columns]

print(y.shape)
print(x.shape)


# Model 1
model = LinearRegression()
model.fit(x,y)

print("Model RSQ: "+str(model.score(x,y)))

print("Coefficients: ")
reg_1 = pd.Series(model.coef_,index=x.columns).sort_values(ascending=False)


# Model 2
model = RandomForestRegressor(max_features=3)
model.fit(x,y)

print("Model Score: "+str(model.score(x,y)))

print("Feature Importance: ")
reg_2 = pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)


# Prediction
pred = pd.Series(model.predict(x),index=x.index)











