
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
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Read preprocessed dataset
bpd_crime_dataframe = pd.read_csv('BPD_Crime_sanitized.csv')


# Remove low frequency premises
threshold = 30 # Anything that occurs less than this will be removed.
A = bpd_crime_dataframe['Premise'].value_counts() # Entire DataFrame 
to_remove = A[A <= threshold].index
bpd_crime_dataframe['Premise'].replace(to_remove, 'Unknown', inplace=True)
bpd_crime_dataframe = bpd_crime_dataframe[bpd_crime_dataframe['Premise'] != 'Unknown']


# Shuffle
bpd_crime_dataframe =  bpd_crime_dataframe.sample(frac=1).reset_index(drop=True)

# set input and prediction output
X = pd.concat([pd.get_dummies(bpd_crime_dataframe['Neighborhood'], prefix = 'N'), pd.get_dummies(bpd_crime_dataframe['District'], prefix = 'DS'), pd.get_dummies(bpd_crime_dataframe['Inside/Outside'], prefix = 'IO'), pd.get_dummies(bpd_crime_dataframe['CrimeDay'], prefix = 'CD'), pd.get_dummies(bpd_crime_dataframe['CrimeCode'], prefix = 'C')], axis = 1)

Y = bpd_crime_dataframe['Premise']


X_train = X[:15000]
Y_train = Y[:15000]
X_test = X[15000:30000]
Y_test = Y[15000:30000]

# Train model
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
	accuracy.append(100*results[1])
	precision.append(100*results[2])
	recall.append(100*results[3])

plt.title("Premise vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("% value")

plt.plot(range(1,total_iterations), accuracy, 'r-', range(1,total_iterations), precision, 'b-', range(1,total_iterations), recall, 'g-')
plt.axis([0, 10, 0, 100])

plt.show()