import numpy as np
import pandas as pd

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

#Load preprocessed dataset
arrest_dataframe = pd.read_csv('BPD_Arrests_sanitized.csv')

#Shuffle
arrest_dataframe =  arrest_dataframe.sample(frac=1).reset_index(drop=True)

X = pd.concat([pd.get_dummies(arrest_dataframe['IncidentOffense'], prefix = 'offns'), pd.get_dummies(arrest_dataframe['NormalizedIncidentLocation'], prefix = 'incLoc'), pd.get_dummies(arrest_dataframe['Charge'], prefix = 'C'), pd.get_dummies(arrest_dataframe['Neighborhood'], prefix = 'N'), pd.get_dummies(arrest_dataframe['District'], prefix = 'D'), pd.get_dummies(arrest_dataframe['Post'], prefix = 'P'), pd.get_dummies(arrest_dataframe['Location 1'], prefix = 'L1')], axis = 1)
Y = arrest_dataframe['Age']

X_train = X.iloc[1:(len(X)/2)]
X_test = X.iloc[(len(X)/2) + 1: len(X)]
Y_train = Y.iloc[1:(len(Y)/2)]
Y_test = Y.iloc[(len(Y)/2) + 1: len(Y)]

X_train = X_train[1:10000]
Y_train = Y_train[1:10000]
X_test = X_test[1:10000]
Y_test = Y_test[1:10000]

#Calculate the logistic regression on this model
log_regression = linear_model.LogisticRegression()
log_regression.fit(X_train, Y_train)
score = log_regression.score(X_test,Y_test)
print 'Score for Logistic Regression: ' + str(score*100) + '%'