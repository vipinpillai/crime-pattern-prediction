import numpy as np
import pandas as pd

seed = 7
np.random.seed(seed)

crime_dataframe = pd.read_csv('BPD_Part_1_Victim_Based_Crime_Data.csv')
crime_dataframe['NormalizedCrimeTime'] = crime_dataframe['CrimeTime'].apply(lambda time: time[:5] if len(time) == 8 else time[0:2] + ':' + time[2:])
crime_dataframe['Inside/Outside'] = crime_dataframe['Inside/Outside'].replace(['Inside', 'Outside'], ['I', 'O'])
crime_dataframe['Inside/Outside'] = crime_dataframe['Inside/Outside'].fillna('Unknown')
crime_dataframe = crime_dataframe[crime_dataframe['Inside/Outside'] != 'Unknown']

#crime_dataframe.drop(['Post','Location 1','Total Incidents'],inplace=True,axis=1)

crime_dataframe = crime_dataframe.drop(196469)
crime_dataframe.drop_duplicates(inplace = True)
crime_dataframe['CrimeHour'] = crime_dataframe['CrimeTime'].apply(lambda time: time[:2])

crime_dataframe['CrimeDay'] = crime_dataframe['CrimeDate'].apply(lambda date: pd.Timestamp(date).weekday_name)
crime_dataframe['NormalizedCrimeDate'] = crime_dataframe['CrimeDate'].apply(lambda date: pd.Timestamp(date))

crime_dataframe['Weapon'] = crime_dataframe['Weapon'].fillna('Unknown')
crime_dataframe = crime_dataframe[crime_dataframe['Weapon'] != 'Unknown']

crime_dataframe['Location'] = crime_dataframe['Location'].fillna('Unknown')
crime_dataframe = crime_dataframe[crime_dataframe['Location'] != 'Unknown']

LocationStr = crime_dataframe['Location']
LocationStr = LocationStr.str.split(' ')
LocationStr = LocationStr.apply(lambda s:s[1:])
crime_dataframe['NormalizedLocation'] = LocationStr.str.join(' ')

crime_dataframe.to_csv('BPD_Crime_sanitized.csv', encoding = 'utf-8')