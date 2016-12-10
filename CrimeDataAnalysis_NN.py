# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers.advanced_activations import LeakyReLU


seed = 7
numpy.random.seed(seed)

crime_dataframe = pd.read_csv('BPD_Part_1_Victim_Based_Crime_Data.csv')
crime_dataframe['NormalizedCrimeTime'] = crime_dataframe['CrimeTime'].apply(lambda time: time[:5] if len(time) == 8 else time[0:2] + ':' + time[2:])
crime_dataframe['Inside/Outside'] = crime_dataframe['Inside/Outside'].replace(['Inside', 'Outside'], ['I', 'O'])
crime_dataframe.drop(['Post','Location 1','Total Incidents'],inplace=True,axis=1)
crime_dataframe.drop_duplicates(inplace = True)
crime_dataframe['CrimeHour'] = crime_dataframe['CrimeTime'].apply(lambda time: time[:2])
crime_dataframe = crime_dataframe.drop(196469)
crime_dataframe['CrimeDay'] = crime_dataframe['CrimeDate'].apply(lambda date: pd.Timestamp(date).weekday_name)
crime_dataframe['NormalizedCrimeDate'] = crime_dataframe['CrimeDate'].apply(lambda date: pd.Timestamp(date))


bpd_crime_dataframe = crime_dataframe.copy()
# bpd_crime_dataframe.drop(['CrimeDate','CrimeTime','NormalizedCrimeTime', 'NormalizedCrimeDate'],inplace=True,axis=1)
# bpd_crime_dataframe['CrimeCode'] = bpd_crime_dataframe['CrimeCode'].astype('category').cat.codes
# bpd_crime_dataframe['Description'] = bpd_crime_dataframe['Description'].astype('category').cat.codes
# bpd_crime_dataframe['Location'] = bpd_crime_dataframe['Location'].astype('category').cat.codes
# bpd_crime_dataframe['Inside/Outside'] = bpd_crime_dataframe['Inside/Outside'].astype('category').cat.codes
# bpd_crime_dataframe['Weapon'] = bpd_crime_dataframe['Weapon'].astype('category').cat.codes
# bpd_crime_dataframe['District'] = bpd_crime_dataframe['District'].astype('category').cat.codes
# bpd_crime_dataframe['Neighborhood'] = bpd_crime_dataframe['Neighborhood'].astype('category').cat.codes
# bpd_crime_dataframe['CrimeDay'] = bpd_crime_dataframe['CrimeDay'].astype('category').cat.codes

#Split DataFrames into Training (Training + Validation) and Test
#train_crime_dataframe = bpd_crime_dataframe.iloc[1:(len(bpd_crime_dataframe)/2)]
#test_crime_dataframe = bpd_crime_dataframe.iloc[(len(bpd_crime_dataframe)/2) + 1: len(bpd_crime_dataframe)]


#X = pd.concat([train_crime_dataframe['Location'], train_crime_dataframe['Inside/Outside'], train_crime_dataframe['Weapon'], train_crime_dataframe['District'], train_crime_dataframe['Neighborhood'], train_crime_dataframe['CrimeHour'], train_crime_dataframe['CrimeDay']], axis=1)
X = pd.concat([pd.get_dummies(bpd_crime_dataframe['Neighborhood'], prefix = 'NB'), pd.get_dummies(bpd_crime_dataframe['District'], prefix = 'DS'), pd.get_dummies(bpd_crime_dataframe['Inside/Outside'], prefix = 'IO'), pd.get_dummies(bpd_crime_dataframe['Weapon'], prefix = 'W'), pd.get_dummies(bpd_crime_dataframe['CrimeHour'], prefix = 'CH'), pd.get_dummies(bpd_crime_dataframe['CrimeDay'], prefix = 'CD')], axis = 1)
Y = bpd_crime_dataframe['CrimeCode']

X = X.sample(frac=1).reset_index(drop=True)
Y = Y.sample(frac=1).reset_index(drop=True)


X_train = X.iloc[1:(len(X)/2)]
X_test = X.iloc[(len(X)/2) + 1: len(X)]
Y_train = Y.iloc[1:(len(Y)/2)]
Y_test = Y.iloc[(len(Y)/2) + 1: len(Y)]


X_train = X_train[:10000]
Y_train = Y_train[:10000]

encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train = encoder.transform(Y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_Y_train)

encoder.fit(Y_test)
encoded_Y_test = encoder.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)


# define baseline model
def baseline_model(input_size, output_size):
    # create model
    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size, init='uniform', activation='relu'))
    #model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(input_size, input_dim=input_size, init='uniform', activation='relu'))
    #model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(input_size, input_dim=input_size, init='uniform', activation='relu'))
    #model.add(LeakyReLU(alpha=0.001))
    #model.add(Dense(7, input_dim=7, init='normal', activation='relu'))
    model.add(Dense(output_size, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

build_fn = baseline_model(len(X_train.columns), len(Y.unique()))
build_fn.fit(X_train, dummy_y_train, batch_size=5, nb_epoch = 10, verbose=1, shuffle=True)
build_fn.evaluate(X_test, dummy_y_test, batch_size=5, verbose=1)

#estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=15, batch_size=5, verbose=1)
#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X.values, dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# X_predict = pd.concat([pd.get_dummies(test_crime_dataframe['Neighborhood'], prefix = 'NB'), pd.get_dummies(test_crime_dataframe['District'], prefix = 'DS'), pd.get_dummies(test_crime_dataframe['Inside/Outside'], prefix = 'IO'), pd.get_dummies(test_crime_dataframe['Weapon'], prefix = 'W'), pd.get_dummies(test_crime_dataframe['CrimeHour'], prefix = 'CH'), pd.get_dummies(test_crime_dataframe['CrimeDay'], prefix = 'CD')], axis = 1)
# Y = train_crime_dataframe['CrimeCode']
