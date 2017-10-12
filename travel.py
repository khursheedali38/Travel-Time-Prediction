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
path = 'yellow_tripdata_2015-01.csv'
df = pd.read_csv(path)
df = df.sample(200000)
df.dropna(inplace=True)
print(len(df))


#writing to file random 200K samples
df.to_csv('random_2015.csv')
df = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_distance']]

print(df.shape)




