#Evaluation of the ArrestLocation feature from BPD Arrests Dataset using Bernoulli Naive Bayes

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


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

#Applying Bernoulli Naive Bayes on this model
model = BernoulliNB()
model.fit(X_train,Y_train)
score = model.score(X_test,Y_test)
print 'Score for Bernoulli Naive Bayes: ' + str(score*100) + '%'


