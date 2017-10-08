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
path = 'C:/Users/Khursheed Ali/Downloads/yellow_tripdata_2017-01.csv'
df = pd.read_csv(path, nrows = 300000)
df.dropna(inplace=True)
print(len(df))

df = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'PULocationID', 'DOLocationID']]

##print(df.shape)

#getting the duration of ride
df['unix_pu'] = df['tpep_pickup_datetime']
df['unix_pu'] = pd.to_datetime(df['unix_pu'])
df['hrs_pu'] = df['unix_pu'].dt.hour
df['day'] = df['unix_pu'].dt.weekday
df['unix_pu'] = (df['unix_pu'] - dt.datetime(1970,1,1)).dt.total_seconds()


df['unix_do'] = df['tpep_dropoff_datetime']
df['unix_do'] = pd.to_datetime(df['unix_do'])
df['hrs_do'] = df['unix_do'].dt.hour
df['unix_do'] = (df['unix_do'] - dt.datetime(1970,1,1)).dt.total_seconds()
df['duration'] = df['unix_do'] - df['unix_pu']

#cleaning up data by removing outliers
df['speed'] = (df['trip_distance'] * 3600)//df['duration']
df = df[ (df['speed'] > 6.0)]
df = df[ (df['speed'] < 140.0)]

df.dropna(inplace=True)

#for tableua export
##writer = pd.ExcelWriter('output.xlsx')
##df.to_excel(writer, 'Sheet1')
##writer.save()


##print(df['day'])

###exploratory analysis/correlation analysis
##plt.scatter(df['trip_distance'],df['duration'])
##plt.show()
##df['pace'] = df['duration']/df['trip_distance']
##print(df['pace'])
##plt.scatter(df['trip_distance'],df['pace'])
##plt.show()
##plt.bar(df['hrs_pu'],df['pace'], align='edge')
##plt.show()
##plt.bar(df['day'],df['pace'], align = 'edge')
##plt.show()
##
##
##df = df[['PULocationID', 'DOLocationID', 'pace', 'hrs_pu', 'day']]
##df['label'] = df['pace']
####df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['PULocationID', 'DOLocationID', 'pace'], how="all")
##df.dropna(inplace = True)
##
##print(df['PULocationID'], df['DOLocationID'])
##
##
##X = np.array(df.drop(['label'], 1))
##y = np.array(df['label'])
##
##
##print(len(X), len(y))
##X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
####clf = LinearRegression()
##clf = RandomForestRegressor(n_jobs=2, random_state=0)
##clf.fit(X_train, y_train)
##accuracy = clf.score(X_test, y_test)
##
##
##print(accuracy)

##print(df)
##print(df.dtypes)
