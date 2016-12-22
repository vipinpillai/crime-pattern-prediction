
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

# Read preprocessed dataset
bpd_crime_dataframe = pd.read_csv('BPD_Crime_sanitized.csv')

# Remove low frequency crime codes
bpd_crime_dataframe = bpd_crime_dataframe.ix[~bpd_crime_dataframe.CrimeCode.isin(['8DO', '3LK', '6K', '8FV', '8CV', '3N', '8GV', '8EV', '3EO', '8I', '8CO', '8BV', '8GO', '3EK', '3LO', '3GO'])]

# Shuffle
bpd_crime_dataframe =  bpd_crime_dataframe.sample(frac=1).reset_index(drop=True)

# set input and prediction output
X = pd.concat([pd.get_dummies(bpd_crime_dataframe['Premise'], prefix = 'P'), pd.get_dummies(bpd_crime_dataframe['Neighborhood'], prefix = 'NL'), pd.get_dummies(bpd_crime_dataframe['District'], prefix = 'DS'), pd.get_dummies(bpd_crime_dataframe['Inside/Outside'], prefix = 'IO'), pd.get_dummies(bpd_crime_dataframe['CrimeHour'], prefix = 'CH'), pd.get_dummies(bpd_crime_dataframe['CrimeDay'], prefix = 'CD')], axis = 1)

Y = bpd_crime_dataframe['CrimeCode']
print str(len(Y.unique()))

X_train = X[:10000]
Y_train = Y[:10000]
X_test = X[10000:20000]
Y_test = Y[10000:20000]

print str(len(Y_train.unique()))
print str(len(Y_test.unique()))

# Train model
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
def baseline_model(input_size, output_size):
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=input_size, init='uniform', bias=True, activation='relu'))
    model.add(Dense(50, init='uniform',  bias=True, activation='relu'))
    model.add(Dense(output_size, init='uniform', bias=True))
    model.add(Activation('softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])
    return model

build_fn = baseline_model(len(X_train.columns), len(Y.unique()))
build_fn.summary()
build_fn.fit(X_train.values, dummy_y_train, batch_size =5, nb_epoch = 5, verbose=1, shuffle=True)
results = build_fn.evaluate(X_test.values, dummy_y_test, verbose=1)
print 'Results: ' + str(results)