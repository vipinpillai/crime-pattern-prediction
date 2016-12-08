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

'''
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
'''

seed = 7
numpy.random.seed(seed)



arrest_dataframe = pd.read_csv('BPD_Arrests.csv')
print len(arrest_dataframe)

arrest_dataframe.head(1)#loc[arrest_dataframe['ArrestDate'] == '10/15/2016']


arrest_dataframe['Arrest'] = arrest_dataframe['Arrest'].fillna(0)
arrest_dataframe['Age'] = arrest_dataframe['Age'].fillna(0)

arrest_dataframe['ArrestDay'] = arrest_dataframe['ArrestDate'].apply(lambda date: pd.Timestamp(date).weekday_name)
arrest_dataframe['NormalizedArrestDate'] = arrest_dataframe['ArrestDate'].apply(lambda date: pd.Timestamp(date))


#remove inconsistencies from ArrestTime like [all time have len either 4 or 5] 9.30, 9:30, 09:30, 09.30  ==> 09.30
arrest_dataframe['ArrestTime'] = arrest_dataframe['ArrestTime'].apply(lambda time: time[0:1] + ':' + time[2:4] if len(time) == 4 else time[0:2] + ':' + time[3:5])
arrest_dataframe['ArrestHour'] = arrest_dataframe['ArrestTime'].apply(lambda time: time[:2])

# arrest_dataframe['IncidentOffense'].unique() = 266
arrest_dataframe['Charge'] = arrest_dataframe['Charge'].fillna('0')
arrest_dataframe['ChargeDescription'] = arrest_dataframe['ChargeDescription'].fillna('Unknown Charge')

arrest_dataframe['District'] = arrest_dataframe['District'].fillna('U')
#if needed convert Northeastern to NE, Central to C and so on...

arrest_dataframe['Neighborhood'] = arrest_dataframe['Neighborhood'].fillna('Unknown')
#arrest_dataframe['Post'] = arrest_dataframe['Post'].fillna(0)

arrest_dataframe.drop(['Post','Location 1'],inplace=True,axis=1)
arrest_dataframe.drop_duplicates(inplace = True)

bpd_arrest_dataframe = arrest_dataframe.copy()
bpd_arrest_dataframe.drop(['ArrestDate', 'ArrestTime', 'NormalizedArrestDate'],inplace=True,axis=1)
bpd_arrest_dataframe['Arrest'] = bpd_arrest_dataframe['Arrest'].astype('category').cat.codes
bpd_arrest_dataframe['Sex'] = bpd_arrest_dataframe['Sex'].astype('category').cat.codes
bpd_arrest_dataframe['Race'] = bpd_arrest_dataframe['Race'].astype('category').cat.codes
bpd_arrest_dataframe['ArrestLocation'] = bpd_arrest_dataframe['ArrestLocation'].astype('category').cat.codes
bpd_arrest_dataframe['IncidentOffense'] = bpd_arrest_dataframe['IncidentOffense'].astype('category').cat.codes
bpd_arrest_dataframe['IncidentLocation'] = bpd_arrest_dataframe['IncidentLocation'].astype('category').cat.codes
bpd_arrest_dataframe['Charge'] = bpd_arrest_dataframe['Charge'].astype('category').cat.codes
bpd_arrest_dataframe['ChargeDescription'] = bpd_arrest_dataframe['ChargeDescription'].astype('category').cat.codes
bpd_arrest_dataframe['District'] = bpd_arrest_dataframe['District'].astype('category').cat.codes
bpd_arrest_dataframe['Neighborhood'] = bpd_arrest_dataframe['Neighborhood'].astype('category').cat.codes


print len(arrest_dataframe)
print len(bpd_arrest_dataframe)
print len(bpd_arrest_dataframe['Neighborhood'].unique())


#Split DataFrames into Training (Training + Validation) and Test
train_arrest_dataframe = bpd_arrest_dataframe.iloc[1:(len(bpd_arrest_dataframe)/2)]
test_arrest_dataframe = bpd_arrest_dataframe.iloc[(len(bpd_arrest_dataframe)/2) + 1: len(bpd_arrest_dataframe)]

X = pd.concat([train_arrest_dataframe['Age'], train_arrest_dataframe['Sex'], train_arrest_dataframe['Race'], train_arrest_dataframe['District']], axis=1)
Y = train_arrest_dataframe['Charge']

#X = X[:10000]
#Y = Y[:10000]

len(train_arrest_dataframe)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

print len(dummy_y)

print encoded_Y
print len(encoded_Y)

uniqueCrimeCodes = Y.unique() # OR crime_dataframe.CrimeCode.unique()
#print uniqueCrimeCodes
print len(uniqueCrimeCodes)

X.values

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    
    '''
    model.add(Dense(7, input_dim = 7, init='normal', activation='relu'))
    model.add(Dense(7, input_dim = 7, init='normal', activation='relu'))
    model.add(Dense(7, input_dim = 7, init='normal', activation='relu'))
    model.add(Dense(7, input_dim = 7, init='normal', activation='relu'))
    '''
    
    model.add(Dense(4, input_dim=4, init='uniform', activation='relu'))
    #model.add(Dense(7, input_dim=7, init='uniform', activation='relu'))
    #model.add(Dense(7, input_dim=7, init='uniform', activation='relu'))
    
    #model.add(Dense(input_dim=7, output_dim = 10, init='normal', activation='relu'))
    #model.add(Dense(input_dim=7, output_dim = 10, init='normal', activation='relu'))
    #model.add(Dense(input_dim=11, output_dim=5, init='normal', activation='softmax'))
    #model.add(Dense(input_dim=5, output_dim=7, init='normal', activation='relu'))
    
    model.add(Dense(len(Y.unique()), init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=5, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X.values, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))