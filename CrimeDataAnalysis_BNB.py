from sklearn.naive_bayes import BernoulliNB
import numpy
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

bpd_crime_dataframe = pd.read_csv('BPD_Crime_sanitized.csv')


bpd_crime_dataframe = bpd_crime_dataframe.ix[~bpd_crime_dataframe.CrimeCode.isin(['8DO', '3LK', '6K', '8FV', '8CV', '3N', '8GV', '8EV', '3EO', '8I', '8CO', '8BV', '8GO', '3EK', '3LO', '3GO'])]

X = pd.concat([pd.get_dummies(bpd_crime_dataframe['Premise'], prefix = 'PM'), pd.get_dummies(bpd_crime_dataframe['Neighborhood'], prefix = 'NB'), pd.get_dummies(bpd_crime_dataframe['District'], prefix = 'DS'), pd.get_dummies(bpd_crime_dataframe['Inside/Outside'], prefix = 'IO'), pd.get_dummies(bpd_crime_dataframe['Weapon'], prefix = 'W'), pd.get_dummies(bpd_crime_dataframe['CrimeHour'], prefix = 'CH'), pd.get_dummies(bpd_crime_dataframe['CrimeDay'], prefix = 'CD')], axis = 1)
Y = bpd_crime_dataframe['CrimeCode']

X_train = X.iloc[1:(len(X)/2)]
X_test = X.iloc[(len(X)/2) + 1: len(X)]
Y_train = Y.iloc[1:(len(Y)/2)]
Y_test = Y.iloc[(len(Y)/2) + 1: len(Y)]

print 'len train : ' + str(len(X_train))
model = BernoulliNB()
model.fit(X_train,Y_train)
score = model.score(X_test,Y_test)
print 'Score for Bernoulli Naive Bayes: ' + str(score*100) + '%'
