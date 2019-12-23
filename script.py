import numpy as np
import pandas as pd


# Load data
df = pd.read_csv('./weatherAUS.csv')

# Remove wasted data
df = df.drop(columns=
            ['Cloud9am', 'Cloud3pm',
            'Evaporation', 'Sunshine',
            'Date', 'RISK_MM', 'Location', 
            'WindDir9am', 'WindDir3pm',
            'WindGustDir'], axis=1)

df = df.dropna(how='any')

# Remove stand-off data
from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
df = df[(z<3).all(axis=1)]

# Change string data to numerical values
df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

# Normalize 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
df_scaled = scaler.fit_transform(df.values)
df = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)

# End of pre-processing data #

# EDA

# from sklearn.feature_selection import SelectKBest, chi2

# X = df.loc[:,df.columns!='RainTomorrow']
# Y = df.loc[['RainTomorrow']]

# print(X)
# print(Y)

