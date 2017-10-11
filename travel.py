import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
style.use('ggplot')

#read the data
path = 'train.csv'
df = pd.read_csv(path)
df.dropna(inplace=True)
print(len(df))

df = df[['pickup_datetime', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_duration']]

print(df.shape)

###getting the duration of ride
##df['unix_pu'] = df['tpep_pickup_datetime']
##df['unix_pu'] = pd.to_datetime(df['unix_pu'])
##df['hrs_pu'] = df['unix_pu'].dt.hour
##df['day'] = df['unix_pu'].dt.weekday
##df['unix_pu'] = (df['unix_pu'] - dt.datetime(1970,1,1)).dt.total_seconds()
##
##
##df['unix_do'] = df['tpep_dropoff_datetime']
##df['unix_do'] = pd.to_datetime(df['unix_do'])
##df['hrs_do'] = df['unix_do'].dt.hour
##df['unix_do'] = (df['unix_do'] - dt.datetime(1970,1,1)).dt.total_seconds()
##df['duration'] = df['unix_do'] - df['unix_pu']
##
###cleaning up data by removing outliers
##df['speed'] = (df['trip_distance'] * 3600)//df['duration']
##df = df[ (df['speed'] > 6.0)]
##df = df[ (df['speed'] < 140.0)]
##
##df.dropna(inplace=True)


