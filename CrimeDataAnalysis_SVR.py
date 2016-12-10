# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

import numpy as np
import pandas as pd
    # from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# from keras.layers.advanced_activations import LeakyReLU
from sklearn.svm import *

seed = 7
np.random.seed(seed)

bpd_crime_dataframe = pd.read_csv('BPD_Crime_sanitized.csv')

bpd_crime_dataframe.drop(['CrimeDate','CrimeTime','NormalizedCrimeTime', 'NormalizedCrimeDate'],inplace=True,axis=1)
bpd_crime_dataframe['CrimeCode'] = bpd_crime_dataframe['CrimeCode'].astype('category').cat.codes
bpd_crime_dataframe['Description'] = bpd_crime_dataframe['Description'].astype('category').cat.codes
bpd_crime_dataframe['Location'] = bpd_crime_dataframe['Location'].astype('category').cat.codes
bpd_crime_dataframe['Inside/Outside'] = bpd_crime_dataframe['Inside/Outside'].astype('category').cat.codes
bpd_crime_dataframe['Weapon'] = bpd_crime_dataframe['Weapon'].astype('category').cat.codes
bpd_crime_dataframe['District'] = bpd_crime_dataframe['District'].astype('category').cat.codes
bpd_crime_dataframe['Neighborhood'] = bpd_crime_dataframe['Neighborhood'].astype('category').cat.codes
bpd_crime_dataframe['CrimeDay'] = bpd_crime_dataframe['CrimeDay'].astype('category').cat.codes

#Split DataFrames into Training (Training + Validation) and Test
#train_crime_dataframe = bpd_crime_dataframe.iloc[1:(len(bpd_crime_dataframe)/2)]
#test_crime_dataframe = bpd_crime_dataframe.iloc[(len(bpd_crime_dataframe)/2) + 1: len(bpd_crime_dataframe)]


#X = pd.concat([train_crime_dataframe['Location'], train_crime_dataframe['Inside/Outside'], train_crime_dataframe['Weapon'], train_crime_dataframe['District'], train_crime_dataframe['Neighborhood'], train_crime_dataframe['CrimeHour'], train_crime_dataframe['CrimeDay']], axis=1)
X = pd.concat([bpd_crime_dataframe['Neighborhood'], bpd_crime_dataframe['District'], bpd_crime_dataframe['Inside/Outside'], bpd_crime_dataframe['Weapon'], bpd_crime_dataframe['CrimeHour'], bpd_crime_dataframe['CrimeDay']], axis = 1)
Y = bpd_crime_dataframe['CrimeCode']

X = X.sample(frac=1).reset_index(drop=True)
Y = Y.sample(frac=1).reset_index(drop=True)


X_train = X.iloc[1:(len(X)/2)]
X_test = X.iloc[(len(X)/2) + 1: len(X)]
Y_train = Y.iloc[1:(len(Y)/2)]
Y_test = Y.iloc[(len(Y)/2) + 1: len(Y)]


X_train = X_train[:10000]
Y_train = Y_train[:10000]

# encoder = LabelEncoder()
# encoder.fit(Y_train)
# encoded_Y_train = encoder.transform(Y_train)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y_train = np_utils.to_categorical(encoded_Y_train)
# encoder.fit(Y_test)
# encoded_Y_test = encoder.transform(Y_test)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y_test = np_utils.to_categorical(encoded_Y_test)

print 'beginning training'
instance = SVC(decision_function_shape='ovr')
instance.fit(X_train, Y_train)
print str(instance.score(X_train, Y_train))



#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# X_predict = pd.concat([pd.get_dummies(test_crime_dataframe['Neighborhood'], prefix = 'NB'), pd.get_dummies(test_crime_dataframe['District'], prefix = 'DS'), pd.get_dummies(test_crime_dataframe['Inside/Outside'], prefix = 'IO'), pd.get_dummies(test_crime_dataframe['Weapon'], prefix = 'W'), pd.get_dummies(test_crime_dataframe['CrimeHour'], prefix = 'CH'), pd.get_dummies(test_crime_dataframe['CrimeDay'], prefix = 'CD')], axis = 1)
# Y = train_crime_dataframe['CrimeCode']
