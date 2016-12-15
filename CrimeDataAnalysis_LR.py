#Baseline for BPD Crime Dataset using Logistic Regression, Accuracy = 51.89%


import numpy
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

bpd_crime_dataframe = pd.read_csv('BPD_Crime_sanitized.csv')

bpd_crime_dataframe = bpd_crime_dataframe.ix[~bpd_crime_dataframe.CrimeCode.isin(['8DO', '3LK', '6K', '8FV', '8CV', '3N', '8GV', '8EV', '3EO', '8I', '8CO', '8BV', '8GO', '3EK', '3LO', '3GO'])]
bpd_crime_dataframe =  bpd_crime_dataframe.sample(frac=1).reset_index(drop=True)

X = pd.concat([pd.get_dummies(bpd_crime_dataframe['Neighborhood'], prefix = 'NB'), pd.get_dummies(bpd_crime_dataframe['District'], prefix = 'DS'), pd.get_dummies(bpd_crime_dataframe['Inside/Outside'], prefix = 'IO'), pd.get_dummies(bpd_crime_dataframe['Weapon'], prefix = 'W'), pd.get_dummies(bpd_crime_dataframe['CrimeHour'], prefix = 'CH'), pd.get_dummies(bpd_crime_dataframe['CrimeDay'], prefix = 'CD')], axis = 1)
Y = bpd_crime_dataframe['CrimeCode']

X_train = X.iloc[1:(len(X)/2)]
X_test = X.iloc[(len(X)/2) + 1: len(X)]
Y_train = Y.iloc[1:(len(Y)/2)]
Y_test = Y.iloc[(len(Y)/2) + 1: len(Y)]

X_train = X_train[1:50000]
Y_train = Y_train[1:50000]
X_test = X_test[1:50000]
Y_test = Y_test[1:50000]


log_regression = linear_model.LogisticRegression()
log_regression.fit(X_train, Y_train)
score = log_regression.score(X_test,Y_test)
print 'Score for Logistic Regression: ' + str(score*100) + '%'
