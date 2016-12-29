import numpy
import pandas as pd

seed = 7
numpy.random.seed(seed)

arrest_dataframe = pd.read_csv('BPD_Arrests.csv')

arrest_dataframe['ArrestDay'] = arrest_dataframe['ArrestDate'].apply(lambda date: pd.Timestamp(date).weekday_name)
arrest_dataframe['NormalizedArrestDate'] = arrest_dataframe['ArrestDate'].apply(lambda date: pd.Timestamp(date))
arrest_dataframe['ArrestTime'] = arrest_dataframe['ArrestTime'].apply(lambda time: time[0:2] + ':' + time[3:5] if len(time) == 5 else time[0:1] + ':' + time[2:4])
arrest_dataframe['ArrestHour'] = arrest_dataframe['ArrestTime'].apply(lambda time: '0' + time[:1] if len(time) == 4 else time[:2])

arrest_dataframe['Arrest'] = arrest_dataframe['Arrest'].fillna('Unknown') #0
arrest_dataframe['Age'] = arrest_dataframe['Age'].fillna('Unknown') #0

arrest_dataframe['ArrestLocation'] = arrest_dataframe['ArrestLocation'].fillna('Unknown')
arrest_dataframe['IncidentOffense'] = arrest_dataframe['IncidentOffense'].fillna('Unknown')
arrest_dataframe['IncidentLocation'] = arrest_dataframe['IncidentLocation'].fillna('Unknown')
arrest_dataframe['Charge'] = arrest_dataframe['Charge'].fillna('Unknown')
arrest_dataframe['ChargeDescription'] = arrest_dataframe['ChargeDescription'].fillna('Unknown')
arrest_dataframe['District'] = arrest_dataframe['District'].fillna('Unknown')
arrest_dataframe['Neighborhood'] = arrest_dataframe['Neighborhood'].fillna('Unknown')
arrest_dataframe['Post'] = arrest_dataframe['Post'].fillna('Unknown')
arrest_dataframe['Location 1'] = arrest_dataframe['Location 1'].fillna('Unknown')
#arrest_dataframe.drop(['Post','Location 1'],inplace=True,axis=1)

arrest_dataframe.drop_duplicates(inplace = True)

arrest_dataframe = arrest_dataframe[arrest_dataframe['Arrest'] != 'Unknown']
arrest_dataframe = arrest_dataframe[arrest_dataframe['Age'] != 'Unknown']
arrest_dataframe = arrest_dataframe[arrest_dataframe['Age'] != 0.0]

arrest_dataframe = arrest_dataframe[arrest_dataframe['IncidentOffense'] != 'Unknown Offense']
arrest_dataframe = arrest_dataframe[arrest_dataframe['IncidentOffense'] != 'UNKNOWN OFFENSE']

arrest_dataframe = arrest_dataframe[arrest_dataframe['IncidentLocation'] != 'Unknown']
arrest_dataframe = arrest_dataframe[arrest_dataframe['ArrestLocation'] != 'Unknown']
arrest_dataframe = arrest_dataframe[arrest_dataframe['Charge'] != 'Unknown']
arrest_dataframe = arrest_dataframe[arrest_dataframe['ChargeDescription'] != 'Unknown']
arrest_dataframe = arrest_dataframe[arrest_dataframe['Post'] != 'Unknown']
arrest_dataframe = arrest_dataframe[arrest_dataframe['Location 1'] != 'Unknown']

ArrestLocationStr = arrest_dataframe['ArrestLocation']
ArrestLocationStr = ArrestLocationStr.str.split(' ')
ArrestLocationStr = ArrestLocationStr.apply(lambda s:s[1:])
arrest_dataframe['NormalizedArrestLocation'] = ArrestLocationStr.str.join(' ')

IncidentLocationStr = arrest_dataframe['IncidentLocation']
IncidentLocationStr = IncidentLocationStr.str.split(' ')
IncidentLocationStr = IncidentLocationStr.apply(lambda s:s[1:])
arrest_dataframe['NormalizedIncidentLocation'] = IncidentLocationStr.str.join(' ')


#Normalize Incident Offences
arrest_dataframe['IncidentOffense'] = arrest_dataframe['IncidentOffense'].\
apply(lambda offns: '102' \
if (offns == '102-Questional Death')
else offns).\
apply(lambda offns: '103' \
if (offns == '103-Dead On Arrival')
else offns).\
apply(lambda offns: '104' \
if (offns == '104-Malicious Burning')
else offns).\
apply(lambda offns: '105' \
if (offns == '105-Suspicious Burning')
else offns).\
apply(lambda offns: '106' \
if (offns == '106-Custody Dispute')
else offns).\
apply(lambda offns: '107' \
if (offns == '107-Drunkenness')
else offns).\
apply(lambda offns: '108' \
if (offns == '108-Liquor Law/Open Containe' or offns == '108-Liquor Law/Open Conta' or \
offns == '108-Liquor Law/Open Contain')
else offns).\
apply(lambda offns: '109' \
if (offns == '109-Loitering')
else offns).\
apply(lambda offns: '110' \
if (offns == '110-Summons Served')
else offns).\
apply(lambda offns: '111' \
if (offns == '111PROTECTIVE ORDER' or offns == '111-Protective Order' or offns == '111-Protective Ord')
else offns).\
apply(lambda offns: '112' \
if (offns == '112-Traffic Related Incident' or offns == '112-Traffic Related Incid' or \
offns == '112-Traffic Relate' or offns == '112-Traffic Related Inc' or offns == '112-Traffic Related Inciden')
else offns).\
apply(lambda offns: '113' \
if (offns == '113-Littering')
else offns).\
apply(lambda offns: '114' \
if (offns == '114-Hindering')
else offns) .\
apply(lambda offns: '115' \
if (offns == '115-Trespassing' or offns == '115TRESPASSING')
else offns).\
apply(lambda offns: '116' \
if (offns == '116-Public Urination / De' or offns == '116-Public Urination / Defe' \
or offns == '116-Public Urination / Defec')
else offns).\
apply(lambda offns: '117' \
if (offns == '117-Fto' or offns == '117FTO')
else offns).\
apply(lambda offns: '118' \
if (offns == '118-Burglary - Fourth Deg' or offns == '118-Burglary - Fourth Degre' \
or offns == '118-Burglary - Fourth Degree' or offns == '118BURGLARY - FOURTH DEGREE')
else offns).\
apply(lambda offns: '119' \
if (offns == '119-Issued In Error' or offns == '119ISSUED IN ERROR')
else offns).\
apply(lambda offns: '1A' \
if (offns == '1A-Murder' or offns == '1AMURDER')
else offns).\
apply(lambda offns: '1B' \
if (offns == '1BMANSLAUGHTER')
else offns).\
apply(lambda offns: '20A' \
if (offns == '20A-Followup')
else offns).\
apply(lambda offns: '20H' \
if (offns == '20H-Traffic Control')
else offns).\
apply(lambda offns: '20J' \
if (offns == '20J-')
else offns).\
apply(lambda offns: '23' \
if (offns == '23UNAUTHORIZED USE' or offns == '23-Unauthorized Use' or offns == '23-Unauthorized Us')
else offns).\
apply(lambda offns: '24' \
if (offns == '24TOWED VEHICLE' or offns == '24-Towed Vehicle')
else offns).\
apply(lambda offns: '24P' \
if (offns == '24P-Towed Vehicle - Private')
else offns).\
apply(lambda offns: '26' \
if (offns == '26RECOVERED VEHICLE' or offns == '26-Recovered Vehicle')
else offns).\
apply(lambda offns: '28' \
if (offns == '28-Suicide - Attempt')
else offns).\
apply(lambda offns: '29' \
if (offns == '29-Driving While Intox.')
else offns).\
apply(lambda offns: '2A' \
if (offns == '2ARAPE (FORCE)' or offns == '2A-Rape (Force)')
else offns).\
apply(lambda offns: '2B' \
if (offns == '2BRAPE (ATTEMPT)' or offns == '2B-Rape (Attempt)')
else offns).\
apply(lambda offns: '2C' \
if (offns == '2C-Carnal Knowledge')
else offns).\
apply(lambda offns: '2D' \
if (offns == '2D-Statutory Rape')
else offns).\
apply(lambda offns: '2F' \
if (offns == '2FPLACING HANDS' or offns == '2F-Placing Hands')
else offns).\
apply(lambda offns: '2G' \
if (offns == '2G-Sodomy/Perverson')
else offns).\
apply(lambda offns: '2H' \
if (offns == '2H-Indecent Exp.')
else offns).\
apply(lambda offns: '2J' \
if (offns == '2JOTHER SEX OFFN.' or offns == '2J-Other Sex Offn.')
else offns).\
apply(lambda offns: '33' \
if (offns == '33-Parking Complaint')
else offns).\
apply(lambda offns: '39' \
if (offns == '39-Fire')
else offns).\
apply(lambda offns: '3AF' \
if (offns == '3AF-Robb Hwy-Firea' or offns == '3AF-Robb Hwy-Firearm' or offns == '3AFROBB HWY-FIREARM')
else offns).\
apply(lambda offns: '3AJF' \
if (offns == '3AJF-Robb Carjack-Firearm' or offns == '3AJFROBB CARJACK-FIREARM')
else offns).\
apply(lambda offns: '3AJK' \
if (offns == '3AJK-Robb Carjack-Knife' or offns == '3AJKROBB CARJACK-KNIFE')
else offns).\
apply(lambda offns: '3AJO' \
if (offns == '3AJO-Robb Carjack-Other Wpn' or offns == '3AJOROBB CARJACK-OTHER WPN')
else offns).\
apply(lambda offns: '3AK' \
if (offns == '3AK-Robb Hwy-Knife' or offns == '3AKROBB HWY-KNIFE')
else offns).\
apply(lambda offns: '3AO' \
if (offns == '3AO-Robb Hwy-Other' or offns == '3AO-Robb Hwy-Other Wpn' or offns == '3AOROBB HWY-OTHER WPN')
else offns).\
apply(lambda offns: '3B' \
if (offns == '3B-Robb Highway (U' or offns == '3B-Robb Highway (Ua)' or offns == '3BROBB HIGHWAY (UA)')
else offns).\
apply(lambda offns: '3BJ' \
if (offns == '3BJ-Robb Carjack(Ua)' or offns == '3BJROBB CARJACK(UA)')
else offns).\
apply(lambda offns: '3CF' \
if (offns == '3CF-Robb Comm-Fire' or offns == '3CF-Robb Comm-Firearm' or offns == '3CFROBB COMM-FIREARM')
else offns).\
apply(lambda offns: '3CK' \
if (offns == '3CK-Robb Comm-Knife' or offns == '3CKROBB COMM-KNIFE')
else offns).\
apply(lambda offns: '3CO' \
if (offns == '3CO-Robb Comm-Other Wpn' or offns == '3COROBB COMM-OTHER WPN')
else offns).\
apply(lambda offns: '3D' \
if (offns == '3D-Robb Comm. (Ua)' or offns == '3DROBB COMM. (UA)')
else offns).\
apply(lambda offns: '3EF' \
if (offns == '3EF-Robb Gas Station-Fi' or offns == '3EF-Robb Gas Station-Firearm')
else offns).\
apply(lambda offns: '3EK' \
if (offns == '3EK-Robb Gas Station-Knife')
else offns).\
apply(lambda offns: '3EO' \
if (offns == '3EO-Robb Gas Station-Other W')
else offns).\
apply(lambda offns: '3F' \
if (offns == '3F-Robb Gas Sta. (Ua)')
else offns).\
apply(lambda offns: '3GF' \
if (offns == '3GF-Robb Conv Store-Firearm')
else offns).\
apply(lambda offns: '3GK' \
if (offns == '3GK-Robb Conv Store-Knife')
else offns).\
apply(lambda offns: '3GO' \
if (offns == '3GO-Robb Conv Store-Other Wp')
else offns).\
apply(lambda offns: '3H' \
if (offns == '3H-Robb Conv. Stor.(Ua)' or offns == '3HROBB CONV. STOR.(UA)')
else offns).\
apply(lambda offns: '3JF' \
if (offns == '3JF-Robb Residence' or offns == '3JF-Robb Residence-Fire' or offns == '3JF-Robb Residence-Firear' or offns == '3JF-Robb Residence-Firearm' or offns == '3JFROBB RESIDENCE-FIREARM')
else offns).\
apply(lambda offns: '3JK' \
if (offns == '3JK-Robb Residence-Knife' or offns == '3JKROBB RESIDENCE-KNIFE')
else offns).\
apply(lambda offns: '3JO' \
if (offns == '3JO-Robb Residence' or offns == '3JO-Robb Residence-Other Wpn' or offns == '3JOROBB RESIDENCE-OTHER WPN')
else offns).\
apply(lambda offns: '3K' \
if (offns == '3K-Robb Res. (Ua)' or offns == '3KROBB RES. (UA)')
else offns).\
apply(lambda offns: '3LF' \
if (offns == '3LF-Robb Bank-Firearm' or offns == '3LFROBB BANK-FIREARM')
else offns).\
apply(lambda offns: '3LO' \
if (offns == '3LO-Robb Bank-Other Wpn')
else offns).\
apply(lambda offns: '3M' \
if (offns == '3M-Robb Bank (Ua)')
else offns).\
apply(lambda offns: '3NF' \
if (offns == '3NF-Robb Misc-Firearm')
else offns).\
apply(lambda offns: '3NK' \
if (offns == '3NK-Robb Misc-Knife')
else offns).\
apply(lambda offns: '3NO' \
if (offns == '3NO-Robb Misc-Other Wpn' or offns == '3NOROBB MISC-OTHER WPN')
else offns).\
apply(lambda offns: '3P' \
if (offns == '3P-Robb Misc. (Ua)' or offns == '3PROBB MISC. (UA)')
else offns).\
apply(lambda offns: '41' \
if (offns == '41-Human Trafficking')
else offns).\
apply(lambda offns: '48' \
if (offns == '48-Involuntary Det' or offns == '48-Involuntary Detentio' or offns == '48-Involuntary Detention' or offns == '48INVOLUNTARY DETENTION')
else offns).\
apply(lambda offns: '49' \
if (offns == '49-Family Disturba' or offns == '49-Family Disturbance' or offns == '49FAMILY DISTURBANCE')
else offns).\
apply(lambda offns: '4A' \
if (offns == '4A-Agg. Asslt.- Gu' or offns == '4A-Agg. Asslt.- Gun' or offns == '4AAGG. ASSLT.- GUN')
else offns).\
apply(lambda offns: '4B' \
if (offns == '4B-Agg. Asslt.- Cu' or offns == '4B-Agg. Asslt.- Cut' or offns == '4BAGG. ASSLT.- CUT')
else offns).\
apply(lambda offns: '4C' \
if (offns == '4C-Agg. Asslt.- Ot' or offns == '4C-Agg. Asslt.- Oth.' or offns == '4CAGG. ASSLT.- OTH.')
else offns).\
apply(lambda offns: '4D' \
if (offns == '4D-Agg. Asslt.- Ha' or offns == '4D-Agg. Asslt.- Hand' or offns == '4DAGG. ASSLT.- HAND')
else offns).\
apply(lambda offns: '4E' \
if (offns == '4E-Common Assault' or offns == '4ECOMMON ASSAULT')
else offns).\
apply(lambda offns: '4F' \
if (offns == '4F-Assault By Thre' or offns == '4F-Assault By Threat' or offns == '4FASSAULT BY THREAT')
else offns).\
apply(lambda offns: '52A' \
if (offns == '52A-Animal Cruelty')
else offns).\
apply(lambda offns: '54' \
if (offns == '54-Armed Person' or offns == '54ARMED PERSON')
else offns).\
apply(lambda offns: '55' \
if (offns == '55-Disorderly Pers' or offns == '55-Disorderly Person') or offns == '55DISORDERLY PERSON'
else offns).\
apply(lambda offns: '55A' \
if (offns == '55A-Prostitution' or offns == '55APROSTITUTION')
else offns).\
apply(lambda offns: '56' \
if (offns == '56-Missing Person')
else offns).\
apply(lambda offns: '58' \
if (offns == '58-Injured Person')
else offns).\
apply(lambda offns: '59' \
if (offns == '59-Intoxicated Person')
else offns).\
apply(lambda offns: '5A' \
if (offns == '5A-Burg. Res. (For' or offns == '5A-Burg. Res. (Force)' or offns == '5ABURG. RES. (FORCE)')
else offns).\
apply(lambda offns: '5B' \
if (offns == '5B-Burg. Res. (Att' or offns == '5B-Burg. Res. (Att.)' or offns == '5BBURG. RES. (ATT.)')
else offns).\
apply(lambda offns: '5C' \
if (offns == '5C-Burg. Res. (Nof' or offns == '5C-Burg. Res. (Noforce)' or offns == '5CBURG. RES. (NOFORCE)')
else offns).\
apply(lambda offns: '5D' \
if (offns == '5D-Burg. Oth. (For' or offns == '5D-Burg. Oth. (Force)' or offns == '5DBURG. OTH. (FORCE)')
else offns).\
apply(lambda offns: '5E' \
if (offns == '5E-Burg. Oth. (Att.)' or offns == '5EBURG. OTH. (ATT.)')
else offns).\
apply(lambda offns: '5F' \
if (offns == '5F-Burg. Oth. (Noforce)' or offns == '5FBURG. OTH. (NOFORCE)')
else offns).\
apply(lambda offns: '60' \
if (offns == '60-Sick Person')
else offns).\
apply(lambda offns: '61' \
if (offns == '61PERSON WANTED ON WAR' or offns == '61-Person Wanted On War')
else offns).\
apply(lambda offns: '67' \
if (offns == '67CHILD ABUSE-PHYSICAL' or offns == '67-Child Abuse-Physical')
else offns).\
apply(lambda offns: '6A' \
if (offns == '6A-Larceny- Pickpocket')
else offns).\
apply(lambda offns: '6B' \
if (offns == '6BLARCENY- PURSE SNATCH' or offns == '6B-Larceny- Purse Snatch' or offns == '6B-Larceny- Purse Snatc')
else offns).\
apply(lambda offns: '6C' \
if (offns == '6CLARCENY- SHOPLIFTING' or offns == '6C-Larceny- Shoplifting' or offns == '6C-Larceny- Shopli')
else offns).\
apply(lambda offns: '6D' \
if (offns == '6DLARCENY- FROM AUTO' or offns == '6D-Larceny- From Auto')
else offns).\
apply(lambda offns: '6E' \
if (offns == '6E-Larceny- Auto Acc' or offns == '6ELARCENY- AUTO ACC')
else offns).\
apply(lambda offns: '6F' \
if (offns == '6F-Larceny- Bicycle' or offns == '6FLARCENY- BICYCLE')
else offns).\
apply(lambda offns: '6G' \
if (offns == '6G-Larceny- From B' or offns == '6G-Larceny- From Bldg.' or offns == '6GLARCENY- FROM BLDG.')
else offns).\
apply(lambda offns: '6H' \
if (offns == '6H-Larceny- From Machine')
else offns).\
apply(lambda offns: '6J' \
if (offns == '6J-Larceny- Other' or offns == '6JLARCENY- OTHER')
else offns).\
apply(lambda offns: '6L' \
if (offns == '6L-Larceny- From Locker')
else offns).\
apply(lambda offns: '70' \
if (offns == '70-Sanitation Complaint')
else offns).\
apply(lambda offns: '70A' \
if (offns == '70A-Ill. Dumping')
else offns).\
apply(lambda offns: '71' \
if (offns == '71-Sex Offender Re' or offns == '71-Sex Offender Registr' or offns == '71-Sex Offender Registry' or offns == '71SEX OFFENDER REGISTRY')
else offns).\
apply(lambda offns: '73' \
if (offns == '73-False Pretense')
else offns).\
apply(lambda offns: '75' \
if (offns == '75-Destruct. Of Pr' or offns == '75-Destruct. Of Propert' or offns == '75-Destruct. Of Property' or offns == '75DESTRUCT. OF PROPERTY')
else offns).\
apply(lambda offns: '76' \
if (offns == '76-Child Abuse-Sexual' or offns == '76CHILD ABUSE-SEXUAL')
else offns).\
apply(lambda offns: '77' \
if (offns == '77-Dog Bite' or offns == '77DOG BITE')
else offns).\
apply(lambda offns: '78' \
if (offns == '78-Gambling' or offns == '78GAMBLING')
else offns).\
apply(lambda offns: '79' \
if (offns == '79-Other' or offns == '79OTHER')
else offns).\
apply(lambda offns: '7A' \
if (offns == '7A-Stolen Auto' or offns == '7ASTOLEN AUTO')
else offns).\
apply(lambda offns: '7C' \
if (offns == '7C-Stolen Veh./Other' or offns == '7CSTOLEN VEH./OTHER')
else offns).\
apply(lambda offns: '80' \
if (offns == '80-Lost Property')
else offns).\
apply(lambda offns: '81' \
if (offns == '81-Recovered Property' or offns == '81RECOVERED PROPERTY')
else offns).\
apply(lambda offns: '83' \
if (offns == '83-Discharging Firearm')
else offns).\
apply(lambda offns: '84' \
if (offns == '84-Bomb Scare')
else offns).\
apply(lambda offns: '85' \
if (offns == '85-Mental Case' or offns == '85MENTAL CASE')
else offns).\
apply(lambda offns: '86' \
if (offns == '86-Special Curfew')
else offns).\
apply(lambda offns: '87' \
if (offns == '87-Narcotics' or offns == '87NARCOTICS')
else offns).\
apply(lambda offns: '87O' \
if (offns == '87O-Narcotics (Out' or offns == '87O-Narcotics (Outside)' or offns == '87ONARCOTICS (OUTSIDE)')
else offns).\
apply(lambda offns: '87V' \
if (offns == '87V-Narcotics (Onview)' or offns == '87VNARCOTICS (ONVIEW)')
else offns).\
apply(lambda offns: '88' \
if (offns == '88-Unfounded Call' or offns == '88UNFOUNDED CALL')
else offns).\
apply(lambda offns: '8AO' \
if (offns == '8AO-Arson Sin Res' or offns == '8AO-Arson Sin Res Str-Occ' or offns == '8AOARSON SIN RES STR-OCC')
else offns).\
apply(lambda offns: '8AV' \
if (offns == '8AV-Arson Sin Res Str-V')
else offns).\
apply(lambda offns: '8BO' \
if (offns == '8BO-Arson Oth Res Str-Occ')
else offns).\
apply(lambda offns: '8EO' \
if (offns == '8EO-Arson Oth Comm Str-Occ')
else offns).\
apply(lambda offns: '8EV' \
if (offns == '8EV-Arson Oth Comm Str-Vac')
else offns).\
apply(lambda offns: '8FO' \
if (offns == '8FO-Arson Public Str-Occ')
else offns).\
apply(lambda offns: '8H' \
if (offns == '8H-Arson Motor Veh')
else offns).\
apply(lambda offns: '8J' \
if (offns == '8J-Arson Other')
else offns).\
apply(lambda offns: '93' \
if (offns == '93-Abduction - Other')
else offns).\
apply(lambda offns: '94' \
if (offns == '94-Abduction By Parent')
else offns).\
apply(lambda offns: '95' \
if (offns == '95-Exparte')
else offns).\
apply(lambda offns: '96' \
if (offns == '96-Investigative Stop' or offns == '96-Stop & Frisk')
else offns).\
apply(lambda offns: '96A' \
if (offns == '96AWEAPONS PAT DOWN')
else offns).\
apply(lambda offns: '96B' \
if (offns == '96B-Investigative Stop' or offns == '96BINVESTIGATIVE STOP')
else offns).\
apply(lambda offns: '97' \
if (offns == '97-Search & Seizur' or offns == '97-Search & Seizure' or offns == '97SEARCH & SEIZURE')
else offns).\
apply(lambda offns: '98' \
if (offns == '98-Child Neglect' or offns == '98CHILD NEGLECT')
else offns)


#Write to file
file_name = 'BPD_Arrests_sanitized.csv'
arrest_dataframe.to_csv(file_name, encoding='utf-8')	  


