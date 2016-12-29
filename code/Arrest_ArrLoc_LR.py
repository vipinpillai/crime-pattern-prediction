#Evaluation of the ArrestLocation feature from BPD Arrests Dataset using Logistic Regression

import numpy as np
import pandas as pd

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

# Load preprocessed dataset
arrest_dataframe = pd.read_csv('../dataset/BPD_Arrests_sanitized.csv')

# Remove low frequency records
threshold = 10 # Anything that occurs less than this will be removed.
A = arrest_dataframe['NormalizedArrestLocation'].value_counts() # Entire DataFrame 
to_remove = A[A <= threshold].index
arrest_dataframe['NormalizedArrestLocation'].replace(to_remove, 'Unknown', inplace=True)
arrest_dataframe = arrest_dataframe[arrest_dataframe['NormalizedArrestLocation'] != 'Unknown']

# Shuffle
arrest_dataframe =  arrest_dataframe.sample(frac=1).reset_index(drop=True)

# One hot encoding of input data
X = pd.concat([pd.get_dummies(arrest_dataframe['IncidentOffense'], prefix = 'offns'), pd.get_dummies(arrest_dataframe['NormalizedIncidentLocation'], prefix = 'NL'), pd.get_dummies(arrest_dataframe['Charge'], prefix = 'C'), pd.get_dummies(arrest_dataframe['Neighborhood'], prefix = 'N'), pd.get_dummies(arrest_dataframe['District'], prefix = 'D')], axis = 1)
Y = arrest_dataframe['NormalizedArrestLocation']

# Split data into train and test
X_train = X.iloc[1:(len(X)/2)]
X_test = X.iloc[(len(X)/2) + 1: len(X)]
Y_train = Y.iloc[1:(len(Y)/2)]
Y_test = Y.iloc[(len(Y)/2) + 1: len(Y)]

#X_train = X_train[1:10000]
#Y_train = Y_train[1:10000]
#X_test = X_test[1:10000]
#Y_test = Y_test[1:10000]

#Calculate the logistic regression on this model
log_regression = linear_model.LogisticRegression()
log_regression.fit(X_train, Y_train)
score = log_regression.score(X_test,Y_test)
print 'Score for Logistic Regression: ' + str(score*100) + '%'