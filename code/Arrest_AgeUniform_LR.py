#Evaluation of the Age feature (Range values) from BPD Arrests Dataset using Logistic Regression 

import numpy as np
import pandas as pd

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

#Load preprocessed dataset
arrest_dataframe = pd.read_csv('BPD_Arrests_sanitized.csv')

#Uniform age
arrest_dataframe['Age'] = arrest_dataframe['Age'].apply(lambda age: 60 if (age >= 60) else age)
arrest_dataframe['Age'] = arrest_dataframe['Age'].apply(lambda age: 50 if (age >= 50 and age < 60) else age)
arrest_dataframe['Age'] = arrest_dataframe['Age'].apply(lambda age: 40 if (age >= 40 and age < 50) else age)
arrest_dataframe['Age'] = arrest_dataframe['Age'].apply(lambda age: 35 if (age >= 35 and age < 40) else age)
arrest_dataframe['Age'] = arrest_dataframe['Age'].apply(lambda age: 30 if (age >= 30 and age < 35) else age)
arrest_dataframe['Age'] = arrest_dataframe['Age'].apply(lambda age: 25 if (age >= 25 and age < 30) else age)
arrest_dataframe['Age'] = arrest_dataframe['Age'].apply(lambda age: 20 if (age >= 20 and age < 25) else age)
arrest_dataframe['Age'] = arrest_dataframe['Age'].apply(lambda age: 15 if (age >= 15 and age < 20) else age)
arrest_dataframe['Age'] = arrest_dataframe['Age'].apply(lambda age: 10 if (age >= 10 and age < 15) else age)

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

#Histogram

A = arrest_dataframe['Age'].sort_values()

plt.hist(A.values)

plt.title(" Histogram")
plt.xlabel("Age Values")
plt.ylabel("Frequency")
plt.show()
