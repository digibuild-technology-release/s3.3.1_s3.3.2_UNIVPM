# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:52:49 2023

@author: utente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import glob
import os


#%%%%CO2
####upload data january to may (june is not provided)
#the path to your csv file directory
mycsvdir1 = 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/co2/TOT'
# get all the csv files in that directory (assuming they have the extension .csv)
csvfiles1 = glob.glob(os.path.join(mycsvdir1, '*.csv'))
# loop through the files and read them in with pandas
dataframes1 = []  # a list to hold all the individual pandas DataFrames
for csvfile1 in csvfiles1:
    df1 = pd.read_csv(csvfile1, sep=';')
    dataframes1.append(df1)
    
# concatenate them all together
result_co2_1_5 = pd.concat(dataframes1, ignore_index=True)
for i1 in range(len(result_co2_1_5)):
        result_co2_1_5['DATE'][i1]=datetime.strptime(result_co2_1_5['DATE'][i1], "%d/%m/%Y %H:%M")
        
#co2 from jen to may
result_co2_1_5=result_co2_1_5.sort_values(by='DATE')

####upload data of july - july has been uploaded eparately since it has a different format for dates
path_july_co2='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/co2/CO2_Luglio2020'
csvfiles_july_co2 = glob.glob(os.path.join(path_july_co2, '*.csv'))
dataframes_july_co2 = []  # a list to hold all the individual pandas DataFrames

for csvfiles_july1 in csvfiles_july_co2:
    df_july_co2 = pd.read_csv(csvfiles_july1, sep=';')
    dataframes_july_co2.append(df_july_co2)
    
# concatenate them all together
result_july_co2 = pd.concat(dataframes_july_co2, ignore_index=True)
for j1 in range(len(result_july_co2)):
        result_july_co2['DATE'][j1]=datetime.strptime(result_july_co2['DATE'][j1], "%Y-%m-%d %H:%M")

#co2 from july
result_july_co2=result_july_co2.sort_values(by='DATE')

####upload data from sept to nov - we need to upload data for this period in a different way since the csv are different from the previous. Moreover, csv file from august is not considered since it has not co2 data
tot1=[]
dataframes_9_11=[]
for root, dirs, files in os.walk("C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/co2/TOT08_12", topdown=False):
        for name in files:
          # print(os.path.join(root, name))
          df2=os.path.join(root,name)
          tot1.append(df2)

for j in range(len(tot1)):
    list_df_co2=pd.read_csv(tot1[j], sep=';')
    matching_columns_co2_1 = list_df_co2.columns[list_df_co2.columns.str.contains('CO2')]
    new_df_co2 = list_df_co2[matching_columns_co2_1].copy()
    new_df_co2.insert(0, 'DATE',list_df_co2['DATE'])
    dataframes_9_11.append(new_df_co2)
    
result_9_11=pd.concat(dataframes_9_11,ignore_index=True)
for k1 in range(len(result_9_11)):
     result_9_11['DATE'][k1]=datetime.strptime(result_9_11['DATE'][k1], "%Y-%m-%d %H:%M")
result_9_11=result_9_11.sort_values(by='DATE')

####upload data of december
path_dec_co2='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/co2/dec'
csvfiles_dec_co2 = glob.glob(os.path.join(path_dec_co2, '*.csv'))
dataframes_co2_12 = []  # a list to hold all the individual pandas DataFrames

for csvfiles_dec2 in csvfiles_dec_co2:
    df_dec_co2= pd.read_csv(csvfiles_dec2, sep=';')
    matching_columns_co2_dec = df_dec_co2.columns[df_dec_co2.columns.str.contains('CO2')]
    new_df_co2_12 = df_dec_co2[matching_columns_co2_dec].copy()
    new_df_co2_12.insert(0, 'DATE',df_dec_co2['DATE'])
    dataframes_co2_12.append(new_df_co2_12)  

# concatenate them all together
result_co2_dec = pd.concat(dataframes_co2_12, ignore_index=True)
result_co2_dec = result_co2_dec.rename(columns=dict(zip(result_co2_dec.columns, result_9_11.columns)))

for h1 in range(len(result_co2_dec)):
        result_co2_dec['DATE'][h1]=datetime.strptime(result_co2_dec['DATE'][h1], "%d/%m/%Y %H:%M")
        
result_co2_dec=result_co2_dec.sort_values(by='DATE')

####concatenate all the co2 file from jen to dec
r_part_co2=[result_co2_1_5, result_july_co2, result_9_11,result_co2_dec]
CO2_jen_dec=pd.concat(r_part_co2, ignore_index=True,axis=0)

####prepare final column co2 
co2_avg=pd.DataFrame()
co2_avg= CO2_jen_dec.iloc[:, 2::3]
col_co2=co2_avg.columns
new_list_co2 =(col_co2.str.split('.').str[1].str.split(' ').str[0]).tolist()
value_to_add_co2=[]

for i2 in range(len(new_list_co2)):
    new_list_value=int(new_list_co2[i2])
    new_value_co2=(np.ones(len(co2_avg)))*new_list_value
    value_to_add_co2.append(new_value_co2)

for ii1, value1 in enumerate(new_list_co2):
    column_name_co2 = f'number sensor{ii1+1}'  # Create a unique column name
    position1 = (ii1) * 2    # Calculate the position to insert the new column
    co2_avg.insert(position1, column_name_co2, value1)
    
c_tot=[]
# concatenate columns 2 by 2
for i in range(0, len(co2_avg.columns), 2):
    column_name = f'Concat_{i//2}'  # Creazione di un nome unico per la colonna concatenata
    concatenated = pd.concat([co2_avg.iloc[:, i], co2_avg.iloc[:, i+1]], axis=1)
    concatenated.columns = ['n_sensor', 'values']  # Rinomina le colonne
    # concatenated_columns.append(column_name)
    c_tot.append(concatenated)
    
co2=pd.concat(c_tot, ignore_index=True)
df_repeated = pd.concat([CO2_jen_dec['DATE']] * 11, ignore_index=True)
co2.insert(0, 'DATE', df_repeated)

#divide the date column into year, month, day, hour, minute
co2['year']=(pd.to_datetime(co2['DATE'])).dt.year
co2['month']=pd.to_datetime(co2['DATE']).dt.month
co2['day']=pd.to_datetime(co2['DATE']).dt.day
co2['hour']=pd.to_datetime(co2['DATE']).dt.hour
co2['minute']=pd.to_datetime(co2['DATE']).dt.minute

#add a columns with the type of data
data_type_co2='indoor CO2'
co2['data_type']=data_type_co2

####save CO2 from jan to dec in a csv
co2.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/focchi_co2_jan_dec_2020.csv')



#%%%%TEMPERATURE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import glob
import os

# the path to your csv file directory
mycsvdir2= 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/indoor_temp/TOT'
# get all the csv files in that directory (assuming they have the extension .csv)
csvfiles2 = glob.glob(os.path.join(mycsvdir2, '*.csv'))
# loop through the files and read them in with pandas
dataframes2 = []  # a list to hold all the individual pandas DataFrames
for csvfile in csvfiles2:
    df2 = pd.read_csv(csvfile, sep=';')
    df2=df2.iloc[:, :52]
    dataframes2.append(df2)
    
# concatenate them all together
result_1_5 = pd.concat(dataframes2, ignore_index=True)
for i3 in range(len(result_1_5)):
        result_1_5['DATE'][i3]=datetime.strptime(result_1_5['DATE'][i3], "%d/%m/%Y %H:%M")
result_1_5=result_1_5.sort_values(by='DATE')

####upload data of july
path_july_temp='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/indoor_temp/Tinterna_Luglio2020'
csvfiles_july_temp = glob.glob(os.path.join(path_july_temp, '*.csv'))
dataframes_july_temp = []  # a list to hold all the individual pandas DataFrames
for csvfiles_july2 in csvfiles_july_temp:
    df_july_temp = pd.read_csv(csvfiles_july2, sep=';')
    df_july_temp=df_july_temp.iloc[:, :52]
    dataframes_july_temp.append(df_july_temp)
    
# concatenate them all together
result_july_temp = pd.concat(dataframes_july_temp, ignore_index=True)

for j2 in range(len(result_july_temp)):
        result_july_temp['DATE'][j2]=datetime.strptime(result_july_temp['DATE'][j2], "%Y-%m-%d %H:%M")
result_july_temp=result_july_temp.sort_values(by='DATE')

####upload data from aug to nov
tot2=[]
dataframes_dec_temp=[]
for root, dirs, files in os.walk("C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/indoor_temp/TOT2", topdown=False):
        for name in files:
          # print(os.path.join(root, name))
          d2=os.path.join(root,name)
          tot2.append(d2)

for k2 in range(len(tot2)):   
    list_df_temp=pd.read_csv(tot2[k2], sep=';')
    new_df_temp = list_df_temp.iloc[:,:52]
    dataframes_dec_temp.append(new_df_temp)
    
result_8_11_temp=pd.concat(dataframes_dec_temp,ignore_index=True)

for h2 in range(len(result_8_11_temp)):
      result_8_11_temp['DATE'][h2]=datetime.strptime(result_8_11_temp['DATE'][h2], "%Y-%m-%d %H:%M")
     
result_8_11_temp=result_8_11_temp.sort_values(by='DATE')
result_8_11_temp=result_8_11_temp.reset_index(drop=True)
result_8_11_temp_res=result_8_11_temp[24:-1]

####upload data of december
path_dec_temp='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/indoor_temp/12_2020'
csvfiles_dec_temp = glob.glob(os.path.join(path_dec_temp, '*.csv'))
dataframes_dec_temp = []  # a list to hold all the individual pandas DataFrames

for csvfiles_dec2 in csvfiles_dec_temp:
    df_dec_temp= pd.read_csv(csvfiles_dec2, sep=';')
    new_df12_temp= df_dec_temp.iloc[:,:52]
    dataframes_dec_temp.append(new_df12_temp)  

# concatenate them all together
result_dec_temp = pd.concat(dataframes_dec_temp, ignore_index=True)
result_dec_temp = result_dec_temp.rename(columns=dict(zip(result_dec_temp.columns, result_8_11_temp.columns)))

for i4 in range(len(result_dec_temp)):
        result_dec_temp['DATE'][i4]=datetime.strptime(result_dec_temp['DATE'][i4], "%d/%m/%Y %H:%M")
        
result_dec_temp=result_dec_temp.sort_values(by='DATE')
result_dec_temp = result_dec_temp.rename(columns=dict(zip(result_dec_temp.columns, new_df_temp.columns)))

####concatenate all the temp file from jan to dec

result_tot_temp=[result_1_5,result_july_temp, result_8_11_temp, result_dec_temp]
temperatures_jan_dec=pd.concat(result_tot_temp, ignore_index=True)

####preparation final columns temperatures
temperatures_avg=pd.DataFrame()
temperatures_avg= temperatures_jan_dec.iloc[:, 2::3]
col_temp=temperatures_avg.columns
new_list_temp=np.arange(1,18,1)
value_to_add_temp=[]

for i5 in range(len(new_list_temp)):
    new_list_value_temp=int(new_list_temp[i5])
    new_value_temp=(np.ones(len(temperatures_avg)))*new_list_value_temp
    value_to_add_temp.append(new_value_temp)

for ii2, value2 in enumerate(new_list_temp):
    column_name_temp = f'number sensor{ii2+1}'  # Create a unique column name
    position2 = (ii2) * 2    # Calculate the position to insert the new column
    temperatures_avg.insert(position2, column_name_temp, value2)

c_tot2=[]
# Concatenazione colonne due a due
for jj in range(0, len(temperatures_avg.columns), 2):
    column_name2 = f'Concat_{jj//2}'  # Creazione di un nome unico per la colonna concatenata
    concatenated_temp = pd.concat([temperatures_avg.iloc[:, jj], temperatures_avg.iloc[:, jj+1]], axis=1)
    concatenated_temp.columns = ['n_sensor', 'values']  # Rinomina le colonne
    c_tot2.append(concatenated_temp)
    
indoor_temp=pd.concat(c_tot2, ignore_index=True)
df_repeated_temp = pd.concat([temperatures_jan_dec['DATE']] * 17, ignore_index=True)

indoor_temp.insert(0, 'DATE', df_repeated_temp)
#divide the date column into year, month, day, hour, minute
indoor_temp['year']=(pd.to_datetime(indoor_temp['DATE'])).dt.year
indoor_temp['month']=(pd.to_datetime(indoor_temp['DATE'])).dt.month
indoor_temp['day']=pd.to_datetime(indoor_temp['DATE']).dt.day
indoor_temp['hour']=pd.to_datetime(indoor_temp['DATE']).dt.hour
indoor_temp['minute']=pd.to_datetime(indoor_temp['DATE']).dt.minute

#add a columns with the type of data
data_type_indoorT='indoor T'
indoor_temp['data_type']=data_type_indoorT

####save indoor temp from jan to dec in a csv
indoor_temp.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/focchi_indoorT_jan_dec_2020.csv')

#%%%%HUMIDITY
# the path to your csv file directory
mycsvdir3 = 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/Indoor_hum/TOT'
# get all the csv files in that directory (assuming they have the extension .csv)
csvfiles3 = glob.glob(os.path.join(mycsvdir3, '*.csv'))
# loop through the files and read them in with pandas
dataframes3 = []  # a list to hold all the individual pandas DataFrames
for csvfile in csvfiles3:
    df3 = pd.read_csv(csvfile, sep=';')
    # df=df.iloc[:, :50]
    dataframes3.append(df3)
      
# concatenate them all together
result_1_5_hum = pd.concat(dataframes3, ignore_index=True)
for i6 in range(len(result_1_5_hum)):
        result_1_5_hum['DATE'][i6]=datetime.strptime(result_1_5_hum['DATE'][i6], "%d/%m/%Y %H:%M")
result_1_5_hum=result_1_5_hum.sort_values(by='DATE')

####upload data of july
path_july_hum='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/Indoor_hum/Uinterna_Luglio2020'
csvfiles_july_hum = glob.glob(os.path.join(path_july_hum, '*.csv'))
dataframes_july_hum = []  # a list to hold all the individual pandas DataFrames
for csvfiles_july3 in csvfiles_july_hum:
    df_july_hum = pd.read_csv(csvfiles_july3, sep=';')
    dataframes_july_hum.append(df_july_hum)
    
# concatenate them all together
result_july_hum = pd.concat(dataframes_july_hum, ignore_index=True)
for j3 in range(len(result_july_hum)):
        result_july_hum['DATE'][j3]=datetime.strptime(result_july_hum['DATE'][j3], "%Y-%m-%d %H:%M")
result_july_hum=result_july_hum.sort_values(by='DATE')

####upload data from aug to nov
tot3=[]
dataframes8_11_hum=[]
for root, dirs, files in os.walk("C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/Indoor_hum/TOT2", topdown=False):
        for name in files:
          # print(os.path.join(root, name))
          d3=os.path.join(root,name)
          tot3.append(d3)

for h3 in range(len(tot3)):
    
    list_df_hum=pd.read_csv(tot3[h3], sep=';')
    matching_columns_hum = list_df_hum.columns[list_df_hum.columns.str.contains('U')]
    new_df_hum = list_df_hum[matching_columns_hum].copy()
    new_df_hum=new_df_hum.iloc[:, :51]
    new_df_hum.insert(0, 'DATE',list_df_hum['DATE'])
    dataframes8_11_hum.append(new_df_hum)
    
result_8_11_hum=pd.concat(dataframes8_11_hum,ignore_index=True)

#sept for humidity has one day (2 sept) that has a different format for the date
result_92=result_8_11_hum[96:120]
result_92=result_92.reset_index(drop=True)
indexes_to_drop=(np.linspace(96,119, 24))
indexes_to_drop=indexes_to_drop.astype(int)
result_8_11_hum_new=result_8_11_hum.drop(indexes_to_drop)
result_8_11_hum_new=result_8_11_hum_new.reset_index(drop=True)

#for loop for the 2 of sept
for k2 in range(len(result_92)):
     result_92['DATE'][k2]=datetime.strptime(result_92['DATE'][k2], "%d/%m/%Y %H:%M")

for kk4 in range(len(result_8_11_hum_new)):
     result_8_11_hum_new['DATE'][kk4]=datetime.strptime(result_8_11_hum_new['DATE'][kk4], "%Y-%m-%d %H:%M") 
     
result_hum_with209=[result_92, result_8_11_hum_new]

result_8_11_hum_final=pd.concat(result_hum_with209) 
result_8_11_hum_final=result_8_11_hum_final.sort_values(by='DATE')

####upload data of december
path_dec_hum='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/Indoor_hum/12_2020'
csvfiles_dec_hum = glob.glob(os.path.join(path_dec_hum, '*.csv'))
dataframes_dec_hum = []  # a list to hold all the individual pandas DataFrames

for csvfiles_dec3 in csvfiles_dec_hum:
    df_dec_hum= pd.read_csv(csvfiles_dec3, sep=';')
    
    matching_columns_dec_hum = df_dec_hum.columns[df_dec_hum.columns.str.contains('U')]
    
    new_df_12_hum = df_dec_hum[matching_columns_dec_hum].copy()
    new_df_12_hum=new_df_12_hum.iloc[:, :51]
    new_df_12_hum.insert(0, 'DATE',df_dec_hum['DATE'])
    dataframes_dec_hum.append(new_df_12_hum)  

# concatenate them all together
result_dec_hum = pd.concat(dataframes_dec_hum, ignore_index=True)
result_dec_hum = result_dec_hum.rename(columns=dict(zip(result_dec_hum.columns, result_8_11_hum_final.columns)))

for i7 in range(len(result_dec_hum)):
        result_dec_hum['DATE'][i7]=datetime.strptime(result_dec_hum['DATE'][i7], "%d/%m/%Y %H:%M")
        
result_dec_hum=result_dec_hum.sort_values(by='DATE')

####concatenate all the temp file from jan to dec
r_part_hum=[result_1_5_hum,result_july_hum,result_8_11_hum_final,result_dec_hum]
humidity_jan_dec=pd.concat(r_part_hum, ignore_index=True)


####preparation final columns humidity
humidity_avg=pd.DataFrame()
humidity_avg= humidity_jan_dec.iloc[:, 2::3]
col_hum=humidity_avg.columns
new_list_hum=np.arange(1,18,1)
value_to_add_hum=[]

for i8 in range(len(new_list_hum)):
    new_list_value_hum=int(new_list_hum[i8])
    new_value_hum=(np.ones(len(humidity_avg)))*new_list_value_hum
    value_to_add_hum.append(new_value_hum)

for ii3, value3 in enumerate(new_list_hum):
    column_name_hum = f'number sensor{ii3+1}'  # Create a unique column name
    position3 = (ii3) * 2    # Calculate the position to insert the new column
    humidity_avg.insert(position3, column_name_hum, value3)
c_tot3=[]
# Concatenazione colonne due a due
for jj2 in range(0, len(humidity_avg.columns), 2):
    column_name_hum= f'Concat_{jj2//2}'  # Creazione di un nome unico per la colonna concatenata
    concatenated_hum = pd.concat([humidity_avg.iloc[:, jj2], humidity_avg.iloc[:, jj2+1]], axis=1)
    concatenated_hum.columns = ['n_sensor', 'values']  # Rinomina le colonne
    # concatenated_columns.append(column_name)
    c_tot3.append(concatenated_hum)

indoor_humidity=pd.concat(c_tot3, ignore_index=True)
df_repeated_hum = pd.concat([humidity_jan_dec['DATE']] * 17, ignore_index=True)
indoor_humidity.insert(0, 'DATE', df_repeated_hum)

#divide the date column into year, month, day, hour, minute
indoor_humidity['year']=(pd.to_datetime(indoor_humidity['DATE'])).dt.year
indoor_humidity['month']=(pd.to_datetime(indoor_humidity['DATE'])).dt.month
indoor_humidity['day']=pd.to_datetime(indoor_humidity['DATE']).dt.day
indoor_humidity['hour']=pd.to_datetime(indoor_humidity['DATE']).dt.hour
indoor_humidity['minute']=pd.to_datetime(indoor_humidity['DATE']).dt.minute

#add a columns with the type of data
data_type_indoorRH='indoor RH'
indoor_humidity['data_type']=data_type_indoorRH

####save indoor rh from jan to dec in a csv
indoor_humidity.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/focchi_indoorRH_jan_dec_2020.csv')


#%%%%OUTDOOR TEMP
# the path to your csv file directory
mycsvdir4 = 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/outdoor_temp/TOT'
# get all the csv files in that directory (assuming they have the extension .csv)
csvfiles4 = glob.glob(os.path.join(mycsvdir4, '*.csv'))
# loop through the files and read them in with pandas
dataframes4 = []  # a list to hold all the individual pandas DataFrames
for csvfile4 in csvfiles4:
    df4 = pd.read_csv(csvfile4, sep=';')
    # df=df.iloc[:, :50]
    dataframes4.append(df4)
      
# concatenate them all together
result_out_1_5 = pd.concat(dataframes4, ignore_index=True)
for i9 in range(len(result_out_1_5)):
        result_out_1_5['DATE'][i9]=datetime.strptime(result_out_1_5['DATE'][i9], "%d/%m/%Y %H:%M")
        
result_out_1_5=result_out_1_5.sort_values(by='DATE')

####upload data of july
path_july_out='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/outdoor_temp/Meteo_Luglio2020'
csvfiles_july_out= glob.glob(os.path.join(path_july_out, '*.csv'))
dataframes_july_out = []  # a list to hold all the individual pandas DataFrames
for csvfile_july_out in csvfiles_july_out:
    df_july_out = pd.read_csv(csvfile_july_out, sep=';')
    # df_july=df_july.iloc[:, :50]
    dataframes_july_out.append(df_july_out)
    
# concatenate them all together
result_july_out = pd.concat(dataframes_july_out, ignore_index=True)

for j4 in range(len(result_july_out)):
        result_july_out['DATE'][j4]=datetime.strptime(result_july_out['DATE'][j4], "%Y-%m-%d %H:%M")
        
result_july_out=result_july_out.sort_values(by='DATE')
r_out=[result_out_1_5, result_july_out]
result_out_1=pd.concat(r_out, ignore_index=True)

matching_columns_out = result_out_1.columns[result_out_1.columns.str.contains('T_ESTERNA')]
result_jan_jul_out = result_out_1[matching_columns_out].copy()
result_jan_jul_out.insert(0, 'DATE',result_out_1['DATE'])

####upload data from aug to nov
tot_out=[]
dataframes_out=[]
for root, dirs, files in os.walk("C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/outdoor_temp/TOT2", topdown=False):
        for name in files:
          # print(os.path.join(root, name))
          d_out=os.path.join(root,name)
          tot_out.append(d_out)

for jj4 in range(len(tot_out)):
    
    list_df_out=pd.read_csv(tot_out[jj4], sep=';')
    matching_columns_out = list_df_out.columns[list_df_out.columns.str.contains('T_ESTERNA')]
    new_df_out = list_df_out[matching_columns_out].copy()
    new_df_out.insert(0, 'DATE',list_df_out['DATE'])
    dataframes_out.append(new_df_out)
    
result_8_11_out=pd.concat(dataframes_out,ignore_index=True)
     
for k5 in range(len(result_8_11_out)):
     result_8_11_out['DATE'][k5]=datetime.strptime(result_8_11_out['DATE'][k5], "%Y-%m-%d %H:%M")

result_8_11_out=result_8_11_out.sort_values(by='DATE')

####upload data of december
path_dec_out='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/original data/outdoor_temp/12_2020'
csvfiles_dec_out = glob.glob(os.path.join(path_dec_out, '*.csv'))
dataframes_dec_out = []  # a list to hold all the individual pandas DataFrames

for csvfile_dec_out in csvfiles_dec_out:
    df_dec_out= pd.read_csv(csvfile_dec_out, sep=';')
    
    matching_columns_dec_out = df_dec_out.columns[df_dec_out.columns.str.contains('T_ESTERNA')]
    
    new_df_12_out = df_dec_out[matching_columns_dec_out].copy()
    new_df_12_out.insert(0, 'DATE',df_dec_out['DATE'])
    dataframes_dec_out.append(new_df_12_out)

# concatenate them all together
result_dec_out = pd.concat(dataframes_dec_out, ignore_index=True)
result_dec_out = result_dec_out.rename(columns=dict(zip(result_dec_out.columns, result_8_11_out.columns)))

for h4 in range(len(result_dec_out)):
        result_dec_out['DATE'][h4]=datetime.strptime(result_dec_out['DATE'][h4], "%d/%m/%Y %H:%M")
        
result_dec_out=result_dec_out.sort_values(by='DATE')

####concatenate all the dataframes from jan to dec
TOT_to_concat=[result_out_1_5, result_out_1,result_8_11_out,result_dec_out]
outdoor_jen_dec=pd.concat(TOT_to_concat, ignore_index=True)
outdoor_avg=pd.DataFrame()
outdoor_avg['DATE']=outdoor_jen_dec['DATE']
outdoor_avg['values']=outdoor_jen_dec['T_ESTERNA AVG']

#divide the date column into year, month, day, hour, minute
outdoor_avg['year']=(pd.to_datetime(outdoor_avg['DATE'])).dt.year
outdoor_avg['month']=(pd.to_datetime(outdoor_avg['DATE'])).dt.month
outdoor_avg['day']=(pd.to_datetime(outdoor_avg['DATE'])).dt.day
outdoor_avg['hour']=(pd.to_datetime(outdoor_avg['DATE'])).dt.hour
outdoor_avg['minute']=(pd.to_datetime(outdoor_avg['DATE'])).dt.minute

#add a columns with the type of data and sens
sens='weather station'
data_type_out='outdoor T'
outdoor_avg['data_type']=data_type_out
# outdoor_avg['n_sensor']=sens
outdoor_avg.insert(1, 'n_sensor',sens)
# outdoor_avg = outdoor_avg.rename(columns=dict(zip(outdoor_avg[''], result_8_11_out.columns)))

####save outdoor temperature from jan to dec in a csv
outdoor_avg.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/focchi_outdoorT_jan_dec_2020.csv')


#%%%%CREATION OF FINAL DATASET

#the path to your csv file directory
mydir = 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm'
# get all the csv files in that directory (assuming they have the extension .csv)
csvfiles_params = glob.glob(os.path.join(mydir, '*.csv'))
# loop through the files and read them in with pandas
dataframes_param = []  # a list to hold all the individual pandas DataFrames
for csv in csvfiles_params:
    df = pd.read_csv(csv, sep=',')
    dataframes_param.append(df)  
# concatenate them all together
final_dataset= pd.concat(dataframes_param, ignore_index=True)
final_dataset=final_dataset.drop(['Unnamed: 0'], axis=1)
# ####save input dataset
final_dataset.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/input dataset/focchi_input_dataset_univpm.csv')

