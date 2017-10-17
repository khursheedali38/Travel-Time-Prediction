import pickle
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from matplotlib import style
style.use('ggplot')

#read data
df = pd.read_csv('post_analysis_data.csv')

#to numpy array
X = np.array(df.drop(['duration'], 1))
y = np.array(df['duration'])

#splitting training and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

#Analysing the required number of trees for RandomForest
a = np.array([[10, 247]])
for i in range(20, 60, 10):
    clf = RandomForestRegressor(n_estimators = i)
    clf.fit(X_train, y_train)
    y_actual = y_test
    y_pred = clf.predict(X_test)
    rms = sqrt(mean_squared_error(y_actual, y_pred))
    a = np.append(a, [[i, rms]], axis = 0)

plt.plot(a[:, 0], a[: 1], linewidth = 2.0)
plt.show()

#fitting the model
clf = RandomForestRegressor(n_estimators = 50)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)



