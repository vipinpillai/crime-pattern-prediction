#Evaluation of the CrimeCode feature from BPD Crime Dataset using Logistic Regression (Includes Weapon as the input feature)

import numpy
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

# Load preprocessed dataset
bpd_crime_dataframe = pd.read_csv('../dataset/BPD_Crime_sanitized.csv')

# Shuffle
bpd_crime_dataframe =  bpd_crime_dataframe.sample(frac=1).reset_index(drop=True)

X = pd.concat([pd.get_dummies(bpd_crime_dataframe['Premise'], prefix = 'P'), pd.get_dummies(bpd_crime_dataframe['Neighborhood'], prefix = 'NL'), pd.get_dummies(bpd_crime_dataframe['District'], prefix = 'DS'), pd.get_dummies(bpd_crime_dataframe['Inside/Outside'], prefix = 'IO'), pd.get_dummies(bpd_crime_dataframe['Weapon'], prefix = 'W'), pd.get_dummies(bpd_crime_dataframe['CrimeHour'], prefix = 'CH'), pd.get_dummies(bpd_crime_dataframe['CrimeDay'], prefix = 'CD')], axis = 1)

Y = bpd_crime_dataframe['CrimeCode']

# Split data
X_train = X.iloc[1:(len(X)/2)]
X_test = X.iloc[(len(X)/2) + 1: len(X)]
Y_train = Y.iloc[1:(len(Y)/2)]
Y_test = Y.iloc[(len(Y)/2) + 1: len(Y)]

X_train = X_train[1:10000]
Y_train = Y_train[1:10000]
X_test = X_test[1:10000]
Y_test = Y_test[1:10000]

# Applying Logistic Regression
log_regression = linear_model.LogisticRegression()
log_regression.fit(X_train, Y_train)
score = log_regression.score(X_test,Y_test)
print 'Score for Logistic Regression: ' + str(score*100) + '%'
