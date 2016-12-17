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
from keras.regularizers import l2
from keras.layers import Dense, Activation


bpd_crime_dataframe = pd.read_csv('BPD_Crime_sanitized.csv')

bpd_crime_dataframe = bpd_crime_dataframe.ix[~bpd_crime_dataframe.CrimeCode.isin(['8DO', '3LK', '6K', '8FV', '8CV', '3N', '8GV', '8EV', '3EO', '8I', '8CO', '8BV', '8GO', '3EK', '3LO', '3GO'])]


bpd_crime_dataframe =  bpd_crime_dataframe.sample(frac=1).reset_index(drop=True)
bpd_crime_dataframe = bpd_crime_dataframe[1:20000]

X = pd.concat([pd.get_dummies(bpd_crime_dataframe['Neighborhood'], prefix = 'NB') ,pd.get_dummies(bpd_crime_dataframe['Weapon'], prefix = 'W'), pd.get_dummies(bpd_crime_dataframe['CrimeHour'], prefix = 'CH'), pd.get_dummies(bpd_crime_dataframe['CrimeDay'], prefix = 'CD')], axis = 1)
Y = bpd_crime_dataframe['CrimeCode']
print str(len(Y.unique()))

X_train = X[:10000]
Y_train = Y[:10000]
X_test = X[10000:20000]
Y_test = Y[10000:20000]

print str(len(Y_train.unique()))
print str(len(Y_test.unique()))

encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train = encoder.transform(Y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_Y_train)
print str(dummy_y_train.shape)


encoder.fit(Y_test)
encoded_Y_test = encoder.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)
print str(dummy_y_test.shape)

# define baseline model
def baseline_model(input_size=47, output_size=29):
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=input_size, init='uniform', bias=True, activation='relu'))
    # model.add(Dense(input_size, input_dim=input_size, init='uniform'))
    # model.add(LeakyReLU(alpha=0.001))
    # model.add(Dense(input_size, init='glorot_uniform', bias = True, activation='relu'))
    # model.add(Dense(100, init='uniform', activation='relu'))
    # model.add(Dense(29, input_dim=input_size, init='uniform',  bias=True, activation = 'relu'))
    # model.add(LeakyReLU(alpha=0.001))
    #model.add(Dense(input_size, input_dim=input_size, init='uniform', activation='relu'))
    # model.add(Dense(input_size, input_dim=input_size, init='uniform'))
    # model.add(LeakyReLU(alpha=0.001))
    #model.add(Dense(7, input_dim=7, init='normal', activation='relu'))
    model.add(Dense(output_size, init='uniform', bias=True))
    model.add(Activation('softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'fmeasure', 'precision', 'recall'])
    return model

# estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=15, batch_size=5, verbose=1)

# seed = 7
# numpy.random.seed(seed)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X_train.values, dummy_y_train, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

build_fn = baseline_model(len(X_train.columns), len(Y.unique()))
build_fn.summary()
build_fn.fit(X_train.values, dummy_y_train, batch_size =5, nb_epoch = 2, verbose=1, shuffle=True)
results = build_fn.evaluate(X_test.values, dummy_y_test, verbose=1)
print 'Results: ' + str(results)