import numpy as np
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

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)

# Load preprocessed dataset
arrest_dataframe = pd.read_csv('BPD_Arrests_sanitized.csv')

# Remove low frequency records
threshold = 10 # Anything that occurs less than this will be removed.
A = arrest_dataframe['IncidentOffense'].value_counts() # Entire DataFrame 
to_remove = A[A <= threshold].index
arrest_dataframe['IncidentOffense'].replace(to_remove, 'Unknown', inplace=True)
arrest_dataframe = arrest_dataframe[arrest_dataframe['IncidentOffense'] != 'Unknown']

# Shuffle
arrest_dataframe =  arrest_dataframe.sample(frac=1).reset_index(drop=True)

X = pd.concat([pd.get_dummies(arrest_dataframe['Age'], prefix = 'A'), pd.get_dummies(arrest_dataframe['Sex'], prefix = 'S'), pd.get_dummies(arrest_dataframe['Neighborhood'], prefix = 'N'), pd.get_dummies(arrest_dataframe['District'], prefix = 'D')], axis = 1)
Y = arrest_dataframe['IncidentOffense']


# Split data into train and test
X_train = X.iloc[1:(len(X)/2)]
X_test = X.iloc[(len(X)/2) + 1: len(X)]
Y_train = Y.iloc[1:(len(Y)/2)]
Y_test = Y.iloc[(len(Y)/2) + 1: len(Y)]

X_train = X_train[1:10000]
Y_train = Y_train[1:10000]
X_test = X_test[1:10000]
Y_test = Y_test[1:10000]

X_cols = len(X.columns)

# One hot encoding on Y
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
    model.add(Dense(100, input_dim=input_size, init='uniform', bias=True, activation='relu'))
    model.add(Dense(50, init='uniform',  bias=True, activation='relu'))
    model.add(Dense(output_size, init='uniform', bias=True))
    model.add(Activation('softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])
    return model

print 'length: ' + str(len(X_train.columns))

# save accuracy results 
accuracy = []
precision = []
recall = []
total_iterations = 11

for i in range(1,total_iterations):
	build_fn = baseline_model(len(X_train.columns), len(Y.unique()))
	build_fn.summary()
	build_fn.fit(X_train.values, dummy_y_train, batch_size=5, nb_epoch=i, verbose=1, shuffle=True)
	results = build_fn.evaluate(X_test.values, dummy_y_test, verbose=1)
	print 'Results: ' + str(results)
	
	accuracy.append(results[1] * 100)
	precision.append(results[2] * 100)
	recall.append(results[3] * 100)

plt.title("IncidentOffense vs Epoch")
plt.xlabel("Iterations")
plt.ylabel("%")
plt.plot(range(1,total_iterations), accuracy, 'r-', range(1,total_iterations), precision, 'b-', range(1,total_iterations), recall, 'g-')
plt.axis([0, 11, 0, 100])
plt.show()