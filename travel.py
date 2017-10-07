import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#read the data
path = 'C:/Users/Khursheed Ali/Downloads/yellow_tripdata_2017-01.csv'
df = pd.read_csv(path, nrows=3000)
df = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID']]

##print(df.shape)

#getting the duration of ride
df['unix_pu'] = df['tpep_pickup_datetime']
df['unix_pu'] = pd.to_datetime(df['unix_pu'])
df['unix_pu'] = (df['unix_pu'] - dt.datetime(1970,1,1)).dt.total_seconds()

df['unix_do'] = df['tpep_dropoff_datetime']
df['unix_do'] = pd.to_datetime(df['unix_do'])
df['unix_do'] = (df['unix_do'] - dt.datetime(1970,1,1)).dt.total_seconds()
df['duration'] = df['unix_do'] - df['unix_pu']

df.dropna(inplace=True)

#exploratory analysis/correlation analysis
plt.scatter(df['trip_distance'],df['duration'])
plt.show()
df['pace'] = df['duration']/df['trip_distance']
plt.scatter(df['trip_distance'],df['pace'])
plt.show()



##print(df)
##print(df.dtypes)

    
