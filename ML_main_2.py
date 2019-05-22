# Import Library
import pandas as pd
import numpy as np
from tqdm import tqdm
from ta import *

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import display

from scipy.signal import argrelextrema
from scipy.cluster import hierarchy
from scipy.spatial import distance

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, Normalizer


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
#outcomes['close_5d'] = data.close.pct_change(periods=-5)  # next 5 day's returns
#outcomes['close_10d'] = data.close.pct_change(periods=-10)  # next 10 day's week's returns
#outcomes['close_20d'] = data.close.pct_change(periods=-20)  # next 20 day's  returns


# Standardize and normalize features
std_scaler = StandardScaler()
features_scaled = std_scaler.fit_transform(features.dropna())
features_df = pd.DataFrame(features_scaled,index=features.dropna().index)
features_df.columns = features.dropna().columns


# Standardize and normalize outcome
outcomes_scaled = std_scaler.fit_transform(outcomes.dropna())
outcomes_df = pd.DataFrame(outcomes_scaled,index=outcomes.dropna().index)
outcomes_df.columns = outcomes.dropna().columns


# Plot features correlation with outcome
corr = features_df.corrwith(outcomes['close_1d'] )
corr.sort_values().plot.barh(color='Blue',title='Strengh of correlation')


# Plot features correlation 
corr_matrix = features_df.corr()
correlations_array = np.asarray(corr_matrix)

linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')

g = sns.clustermap(corr_matrix, row_linkage=linkage, col_linkage=linkage, row_cluster=True, col_cluster=True, figsize=(10,10), cmap='Greens')

plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()

label_order = corr_matrix.iloc[:,g.dendrogram_row.reordered_ind].columns

print("Correlation Strengh:")
print(corr[corr>0.1].sort_values(ascending=False))


selected_features = ['momentum_rsi','trend_macd_diff','volume_cmf']
sns.pairplot(features_df[selected_features],size=1.5)


tmp = features_df[selected_features].join(outcomes_df).reset_index().set_index('date')
tmp.dropna().resample('M').apply(lambda x: x.corr()).iloc[:,-1].unstack().iloc[:,:-1].plot(title='Correlation of features to outcome by months')









