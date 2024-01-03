# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:25:50 2023

@author: utente
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from datetime import datetime
import glob
import os
from scipy import signal
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import utils
from sPMV_v1 import sPMV_calculation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time

#%%%% Labeling 
####import the final dataset
dataset_dir='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/input dataset/focchi_input_dataset_univpm_v2.csv'
dataset=pd.read_csv(dataset_dir, sep=',')
dataset=dataset.drop(['Unnamed: 0'], axis=1)
dataset['DATE']=pd.to_datetime(dataset['DATE'], format = "%Y-%m-%d %H:%M:%S")

#select from the overall dataset all the information of the loaded dataset that has data_type == 'indoor T' and  data_type == 'indoor RH'
indoorT=dataset.loc[dataset['data_type']=='indoor T'] 
indoorRH=dataset.loc[dataset['data_type']=='indoor RH']
CO2=dataset.loc[(dataset['data_type']=='indoor CO2')]
lux=dataset.loc[(dataset['data_type']=='lux')] #esterni
# filtered_lux = lux[(lux ['values']>= 100)& (lux ['values']<= 400)]
outT=dataset.loc[(dataset['data_type']=='outdoor T')]
outT_filt=outT.loc[(outT['values']>=-10)&(outT['values']<=40)].reset_index(drop=True)
filtered_temperatures=outT_filt[3391:10597]

#find the indexes of Nan and remove
#indoorT and indoorRH
indoorT=indoorT.reset_index(drop=True)
indoorRH=indoorRH.reset_index(drop=True) 
index_RH_nan = indoorRH[indoorRH.isna().any(axis=1)].index
index_T_nan = indoorT[indoorT.isna().any(axis=1)].index
indoorT=indoorT.drop(index_RH_nan)
indoorRH=indoorRH.drop(index_T_nan)
indoorT=indoorT.dropna()
indoorRH=indoorRH.dropna()
indoorT=indoorT.reset_index(drop=True)
indoorRH=indoorRH.reset_index(drop=True)

#CO2
CO2=CO2.reset_index(drop=True)
nan_indicesCO2 = CO2[CO2.isna().any(axis=1)].index
co2_filt=CO2.drop(nan_indicesCO2)
co2_filt=co2_filt.dropna()
co2_filt=co2_filt.reset_index(drop=True)

#lux in datasetOS needs Nan vakues removal
lux=lux.reset_index(drop=True)
nan_indiceslux = lux[lux.isna().any(axis=1)].index
lux_filt=lux.drop(nan_indiceslux)
lux_filt=lux_filt.dropna()
lux_filt=lux_filt.reset_index(drop=True)

#outT
filtered_temperatures=filtered_temperatures.reset_index(drop=True)
nan_indicesoutT = filtered_temperatures[filtered_temperatures.isna().any(axis=1)].index
outT_filt=filtered_temperatures.drop(nan_indicesoutT)
outT_filt=outT_filt.dropna()
outT_filt=outT_filt.reset_index(drop=True)

# indoorT=indoorT.reset_index(drop=True)
# indoorRH=indoorRH.reset_index(drop=True) 
# CO2_fin=co2_filt.reset_index(drop=True) 
# lux_fin=lux.reset_index(drop=True) 
# outT_fin=filtered_temperatures.reset_index(drop=True) 

####openspace
lux_filt['n_sensor']=lux_filt['n_sensor'].astype(float)
datasetOS=pd.DataFrame()
datasetOS['date']=indoorT['DATE'].loc[(indoorT['n_sensor']==6)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetOS['indoorT_OS']=indoorT['values'].loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetOS['indoorRH_OS']=indoorRH['values'].loc[(indoorRH['n_sensor']==6)].reset_index(drop=True)
datasetOS['CO2_OS']=co2_filt['values'].loc[(co2_filt['n_sensor']==31)].reset_index(drop=True) #NAN
datasetOS['lux_OS']=lux_filt['values'].loc[(lux_filt['n_sensor']==2)].reset_index(drop=True)
datasetOS['outT']=outT_filt['values']#filtered_temperatures['values'] #NAN

# Check if there are any NaN values in the entire DataFrame
# has_nan = datasetOS['CO2_OS'].isna().any().any()

#resample of RH, lux, CO2 and outT
#CO2
new_indices_CO2 = np.linspace(0, len(co2_filt) - 1, len(datasetOS['indoorT_OS']))
old_indices_CO2 = np.arange(len(co2_filt))
f1 = interpolate.interp1d(old_indices_CO2, co2_filt['values'], kind='linear')
CO2_res = f1(new_indices_CO2)
nan_indicesc = np.isnan(np.sum(CO2_res))

# #lux
# new_indices_lux = np.linspace(0, len(datasetOS['lux_OS']) - 1, len(datasetOS['indoorT_OS']))
# old_indices_lux = np.arange(len(datasetOS['lux_OS']))
# f2 = interpolate.interp1d(old_indices_lux, datasetOS['lux_OS'], kind='linear',fill_value='extrapolate')
# lux_res = f2(new_indices_lux)
# nan_indicesLR = np.isnan(np.sum(datasetOS['lux_OS']))

#outT
new_indices_outT = np.linspace(0, len(outT_filt) - 1, len(datasetOS['indoorT_OS']))
old_indices_outT = np.arange(len(outT_filt))
f3 = interpolate.interp1d(old_indices_outT,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res = f3(new_indices_outT)
nan_indicesOR = np.isnan(np.sum(outT_res))

#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor=indoorT['n_sensor'].loc[(indoorT['n_sensor']==6)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor=sensor.reset_index(drop=True)
date=indoorT['DATE'].loc[(indoorT['n_sensor']==6)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date=date.reset_index(drop=True)  
date=pd.to_datetime(date, format = "%Y-%m-%d %H:%M:%S")

#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot=sPMV_calculation(datasetOS['indoorT_OS'], datasetOS['indoorRH_OS'],sensor,date) # 
spmv_tot=spmv_tot.reset_index(drop=True)

#create a dataset with the features we will include in the input dataframe
X=pd.DataFrame()
X['date']=datasetOS['date']
X['indoor T']=datasetOS['indoorT_OS']
X['indoor RH']=datasetOS['indoorRH_OS']
X['CO2']=CO2_res
X['outdoorT']=outT_res
X['lux']=datasetOS['lux_OS']
X['sPMV']=spmv_tot['sPMV']
X['room']='openspace'



####sala riunioni 2
lux_filt['n_sensor']=lux_filt['n_sensor'].astype(float)
datasetSR2=pd.DataFrame()
datasetSR2['date']=indoorT['DATE'].loc[(indoorT['n_sensor']==9)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetSR2['indoorT_SR2']=indoorT['values'].loc[(indoorT['n_sensor']==9)].reset_index(drop=True)
datasetSR2['indoorRH_SR2']=indoorRH['values'].loc[(indoorRH['n_sensor']==9)].reset_index(drop=True)
datasetSR2['CO2_SR2']=co2_filt['values'].loc[(co2_filt['n_sensor']==33)].reset_index(drop=True) #NAN
datasetSR2['lux_SR2']=lux_filt['values'].loc[(lux_filt['n_sensor']==8)].reset_index(drop=True)
datasetSR2['outT']=outT_filt['values']#filtered_temperatures['values'] #NAN
#resample of RH, lux, CO2 and outT
# from scipy import interpolate
#CO2
new_indices_CO2_SR2 = np.linspace(0, len(co2_filt) - 1, len(datasetSR2['indoorT_SR2']))
old_indices_CO2_SR2 = np.arange(len(co2_filt))
f1_SR2 = interpolate.interp1d(old_indices_CO2_SR2, co2_filt['values'], kind='linear')
CO2_res_SR2 = f1_SR2(new_indices_CO2_SR2)
nan_indicesc_SR2 = np.isnan(np.sum(CO2_res_SR2))

#outT
new_indices_outT_SR2 = np.linspace(0, len(outT_filt) - 1, len(datasetSR2['indoorT_SR2']))
old_indices_outT_SR2 = np.arange(len(outT_filt))
f3_SR2 = interpolate.interp1d(old_indices_outT_SR2,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res_SR2 = f3_SR2(new_indices_outT_SR2)
nan_indicesOR_SR2 = np.isnan(np.sum(outT_res_SR2))

#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor_SR2=indoorT['n_sensor'].loc[(indoorT['n_sensor']==9)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor_SR2=sensor_SR2.reset_index(drop=True)
date_SR2=indoorT['DATE'].loc[(indoorT['n_sensor']==9)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date_SR2=date_SR2.reset_index(drop=True)  
date_SR2=pd.to_datetime(date_SR2, format = "%Y-%m-%d %H:%M:%S")

#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot_SR2=sPMV_calculation(datasetSR2['indoorT_SR2'], datasetSR2['indoorRH_SR2'],sensor_SR2,date_SR2) # 
spmv_tot_SR2=spmv_tot_SR2.reset_index(drop=True)

#create a dataset with the features we will include in the input dataframe
X_SR2=pd.DataFrame()
X_SR2['date']=datasetSR2['date']
X_SR2['indoor T']=datasetSR2['indoorT_SR2']
X_SR2['indoor RH']=datasetSR2['indoorRH_SR2']
X_SR2['CO2']=CO2_res_SR2
X_SR2['outdoorT']=outT_res_SR2
X_SR2['lux']=datasetSR2['lux_SR2']
X_SR2['sPMV']=spmv_tot_SR2['sPMV']
X_SR2['room']='riunioni2'

####ufficio acquisti A
lux_filt['n_sensor']=lux_filt['n_sensor'].astype(float)
datasetUA_A=pd.DataFrame()
datasetUA_A['date']=indoorT['DATE'].loc[(indoorT['n_sensor']==11)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetUA_A['indoorT_UA_A']=indoorT['values'].loc[(indoorT['n_sensor']==11)].reset_index(drop=True)
datasetUA_A['indoorRH_UA_A']=indoorRH['values'].loc[(indoorRH['n_sensor']==11)].reset_index(drop=True)
datasetUA_A['CO2_UA_A']=co2_filt['values'].loc[(co2_filt['n_sensor']==35)].reset_index(drop=True) #NAN
datasetUA_A['lux_UA_A']=lux_filt['values'].loc[(lux_filt['n_sensor']==8)].reset_index(drop=True)
datasetUA_A['outT']=outT_filt['values']#filtered_temperatures['values'] #NAN

#resample of RH, lux, CO2 and outT
#CO2
new_indices_CO2_UA_A = np.linspace(0, len(co2_filt) - 1, len(datasetUA_A['indoorT_UA_A']))
old_indices_CO2_UA_A = np.arange(len(co2_filt))
f1_UA_A = interpolate.interp1d(old_indices_CO2_UA_A, co2_filt['values'], kind='linear')
CO2_res_UA_A = f1_UA_A(new_indices_CO2_UA_A)
nan_indicesc_UA_A = np.isnan(np.sum(CO2_res_UA_A))

#outT
new_indices_outT_UA_A = np.linspace(0, len(outT_filt) - 1, len(datasetUA_A['indoorT_UA_A']))
old_indices_outT_UA_A = np.arange(len(outT_filt))
f3_UA_A = interpolate.interp1d(old_indices_outT_UA_A,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res_UA_A = f3_UA_A(new_indices_outT_UA_A)
nan_indicesOR_UA_A = np.isnan(np.sum(outT_res_UA_A))

#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor_UA_A=indoorT['n_sensor'].loc[(indoorT['n_sensor']==11)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor_UA_A=sensor_UA_A.reset_index(drop=True)
date_UA_A=indoorT['DATE'].loc[(indoorT['n_sensor']==11)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date_UA_A=date_UA_A.reset_index(drop=True)  
date_UA_A=pd.to_datetime(date_UA_A, format = "%Y-%m-%d %H:%M:%S")

#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot_UA_A=sPMV_calculation(datasetUA_A['indoorT_UA_A'], datasetUA_A['indoorRH_UA_A'],sensor_UA_A,date_UA_A) # 
spmv_tot_UA_A=spmv_tot_UA_A.reset_index(drop=True)

####create a dataset with the features we will include in the input dataframe
X_UA_A=pd.DataFrame()
X_UA_A['date']=datasetUA_A['date']
X_UA_A['indoor T']=datasetUA_A['indoorT_UA_A']
X_UA_A['indoor RH']=datasetUA_A['indoorRH_UA_A']
X_UA_A['CO2']=CO2_res_UA_A
X_UA_A['outdoorT']=outT_res_UA_A
X_UA_A['lux']=datasetUA_A['lux_UA_A']
X_UA_A['sPMV']=spmv_tot_UA_A['sPMV']
X_UA_A['room']='ufficio_a_A'


####ufficio acquisti B
lux_filt['n_sensor']=lux_filt['n_sensor'].astype(float)
datasetUA_B=pd.DataFrame()
datasetUA_B['date']=indoorT['DATE'].loc[(indoorT['n_sensor']==12)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetUA_B['indoorT_UA_B']=indoorT['values'].loc[(indoorT['n_sensor']==12)].reset_index(drop=True)
datasetUA_B['indoorRH_UA_B']=indoorRH['values'].loc[(indoorRH['n_sensor']==12)].reset_index(drop=True)
datasetUA_B['CO2_UA_B']=co2_filt['values'].loc[(co2_filt['n_sensor']==36)].reset_index(drop=True) #NAN
datasetUA_B['lux_UA_B']=lux_filt['values'].loc[(lux_filt['n_sensor']==8)].reset_index(drop=True)
datasetUA_B['outT']=outT_filt['values']#filtered_temperatures['values'] #NAN

#resample of RH, lux, CO2 and outT
from scipy import interpolate
#CO2
new_indices_CO2_UA_B = np.linspace(0, len(co2_filt) - 1, len(datasetUA_B['indoorT_UA_B']))
old_indices_CO2_UA_B = np.arange(len(co2_filt))
f1_UA_B = interpolate.interp1d(old_indices_CO2_UA_B, co2_filt['values'], kind='linear')
CO2_res_UA_B = f1_UA_B(new_indices_CO2_UA_B)
nan_indicesc_UA_B = np.isnan(np.sum(CO2_res_UA_B))

#outT
new_indices_outT_UA_B = np.linspace(0, len(outT_filt) - 1, len(datasetUA_B['indoorT_UA_B']))
old_indices_outT_UA_B = np.arange(len(outT_filt))
f3_UA_B = interpolate.interp1d(old_indices_outT_UA_B,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res_UA_B = f3_UA_B(new_indices_outT_UA_B)
nan_indicesOR_UA_B = np.isnan(np.sum(outT_res_UA_B))

#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor_UA_B=indoorT['n_sensor'].loc[(indoorT['n_sensor']==12)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor_UA_B=sensor_UA_B.reset_index(drop=True)
date_UA_B=indoorT['DATE'].loc[(indoorT['n_sensor']==12)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date_UA_B=date_UA_B.reset_index(drop=True)  
date_UA_B=pd.to_datetime(date_UA_B, format = "%Y-%m-%d %H:%M:%S")

#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot_UA_B=sPMV_calculation(datasetUA_B['indoorT_UA_B'], datasetUA_B['indoorRH_UA_B'],sensor_UA_B,date_UA_B) # 
spmv_tot_UA_B=spmv_tot_UA_B.reset_index(drop=True)

#create a dataset with the features we will include in the input dataframe
X_UA_B=pd.DataFrame()
X_UA_B['date']=datasetUA_B['date']
X_UA_B['indoor T']=datasetUA_B['indoorT_UA_B']
X_UA_B['indoor RH']=datasetUA_B['indoorRH_UA_B']
X_UA_B['CO2']=CO2_res_UA_B
X_UA_B['outdoorT']=outT_res_UA_B
X_UA_B['lux']=datasetUA_B['lux_UA_B']
X_UA_B['sPMV']=spmv_tot_UA_B['sPMV']
X_UA_B['room']='ufficio_a_B'


####sala riunioni 3
lux_filt['n_sensor']=lux_filt['n_sensor'].astype(float)
datasetSR3=pd.DataFrame()
datasetSR3['date']=indoorT['DATE'].loc[(indoorT['n_sensor']==14)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetSR3['indoorT_SR3']=indoorT['values'].loc[(indoorT['n_sensor']==14)].reset_index(drop=True)
datasetSR3['indoorRH_SR3']=indoorRH['values'].loc[(indoorRH['n_sensor']==14)].reset_index(drop=True)
datasetSR3['CO2_SR3']=co2_filt['values'].loc[(co2_filt['n_sensor']==37)].reset_index(drop=True) #NAN
datasetSR3['lux_SR3']=lux_filt['values'].loc[(lux_filt['n_sensor']==8)].reset_index(drop=True)
datasetSR3['outTSR3']=outT_filt['values']#filtered_temperatures['values'] #NAN
#resample of RH, lux, CO2 and outT
#CO2
new_indices_CO2_SR3 = np.linspace(0, len(co2_filt) - 1, len(datasetSR3['indoorT_SR3']))
old_indices_CO2_SR3 = np.arange(len(co2_filt))
f1_SR3 = interpolate.interp1d(old_indices_CO2_SR3, co2_filt['values'], kind='linear')
CO2_res_SR3 = f1_SR3(new_indices_CO2_SR3)
nan_indicesc_SR3 = np.isnan(np.sum(CO2_res_SR3))

#outT
new_indices_outT_SR3 = np.linspace(0, len(outT_filt) - 1, len(datasetSR3['indoorT_SR3']))
old_indices_outT_SR3 = np.arange(len(outT_filt))
f3_SR3 = interpolate.interp1d(old_indices_outT_SR3,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res_SR3 = f3_SR3(new_indices_outT_SR3)
nan_indicesOR_SR3 = np.isnan(np.sum(outT_res_SR3))

#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor_SR3=indoorT['n_sensor'].loc[(indoorT['n_sensor']==14)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor_SR3=sensor_SR3.reset_index(drop=True)
date_SR3=indoorT['DATE'].loc[(indoorT['n_sensor']==14)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date_SR3=date_SR3.reset_index(drop=True)  
date_SR3=pd.to_datetime(date_SR3, format = "%Y-%m-%d %H:%M:%S")
#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot_SR3=sPMV_calculation(datasetSR3['indoorT_SR3'], datasetSR3['indoorRH_SR3'],sensor_SR3,date_SR3) # 
spmv_tot_SR3=spmv_tot_SR3.reset_index(drop=True)

#create a dataset with the features we will include in the input dataframe
X_SR3=pd.DataFrame()
X_SR3['date']=datasetSR3['date']
X_SR3['indoor T']=datasetSR3['indoorT_SR3']
X_SR3['indoor RH']=datasetSR3['indoorRH_SR3']
X_SR3['CO2']=CO2_res_SR3
X_SR3['outdoorT']=outT_res_SR3
X_SR3['lux']=datasetSR3['lux_SR3']
X_SR3['sPMV']=spmv_tot_SR3['sPMV']
X_SR3['room']='riunioni3'

####ufficio qualità A
lux_filt['n_sensor']=lux_filt['n_sensor'].astype(float)
datasetUQ_A=pd.DataFrame()
datasetUQ_A['date']=indoorT['DATE'].loc[(indoorT['n_sensor']==15)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetUQ_A['indoorT_UQ_A']=indoorT['values'].loc[(indoorT['n_sensor']==15)].reset_index(drop=True)
datasetUQ_A['indoorRH_UQ_A']=indoorRH['values'].loc[(indoorRH['n_sensor']==15)].reset_index(drop=True)
datasetUQ_A['CO2_UQ_A']=co2_filt['values'].loc[(co2_filt['n_sensor']==38)].reset_index(drop=True) #NAN
datasetUQ_A['lux_UQ_A']=lux_filt['values'].loc[(lux_filt['n_sensor']==7)].reset_index(drop=True)
datasetUQ_A['outT']=outT_filt['values'] #filtered_temperatures['values'] #NAN

#resample of RH, lux, CO2 and outT
#CO2
new_indices_CO2_UQ_A = np.linspace(0, len(co2_filt) - 1, len(datasetUQ_A['indoorT_UQ_A']))
old_indices_CO2_UQ_A = np.arange(len(co2_filt))
f1_UQ_A = interpolate.interp1d(old_indices_CO2_UQ_A, co2_filt['values'], kind='linear')
CO2_res_UQ_A = f1_UQ_A(new_indices_CO2_UQ_A)
nan_indicesc_UQ_A = np.isnan(np.sum(CO2_res_UQ_A))

#outT
new_indices_outT_UQ_A = np.linspace(0, len(outT_filt) - 1, len(datasetUQ_A['indoorT_UQ_A']))
old_indices_outT_UQ_A = np.arange(len(outT_filt))
f3_UQ_A = interpolate.interp1d(old_indices_outT_UQ_A,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res_UQ_A = f3_UQ_A(new_indices_outT_UQ_A)
nan_indicesOR_UQ_A = np.isnan(np.sum(outT_res_UQ_A))
#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor_UQ_A=indoorT['n_sensor'].loc[(indoorT['n_sensor']==15)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor_UQ_A=sensor_UQ_A.reset_index(drop=True)
date_UQ_A=indoorT['DATE'].loc[(indoorT['n_sensor']==15)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date_UQ_A=date_UQ_A.reset_index(drop=True)  
date_UQ_A=pd.to_datetime(date_UQ_A, format = "%Y-%m-%d %H:%M:%S")
#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot_UQ_A=sPMV_calculation(datasetUQ_A['indoorT_UQ_A'], datasetUQ_A['indoorRH_UQ_A'],sensor_UQ_A,date_UQ_A) # 
spmv_tot_UQ_A=spmv_tot_UQ_A.reset_index(drop=True)
#create a dataset with the features we will include in the input dataframe
X_UQ_A=pd.DataFrame()
X_UQ_A['date']=datasetUQ_A['date']
X_UQ_A['indoor T']=datasetUQ_A['indoorT_UQ_A']
X_UQ_A['indoor RH']=datasetUQ_A['indoorRH_UQ_A']
X_UQ_A['CO2']=CO2_res_UQ_A
X_UQ_A['outdoorT']=outT_res_UQ_A
X_UQ_A['lux']=datasetUQ_A['lux_UQ_A']
X_UQ_A['sPMV']=spmv_tot_UQ_A['sPMV']
X_UQ_A['room']='ufficio_q_A'


####Direzione Ufficio Acquisti
datasetdir=pd.DataFrame()
datasetdir['date']=indoorT['DATE'].loc[(indoorT['n_sensor']==17)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetdir['indoorT']=indoorT['values'].loc[(indoorT['n_sensor']==17)].reset_index(drop=True)
datasetdir['indoorRH']=indoorRH['values'].loc[(indoorRH['n_sensor']==17)].reset_index(drop=True)
datasetdir['CO2']=co2_filt['values'].loc[(co2_filt['n_sensor']==40)].reset_index(drop=True) #NAN
datasetdir['lux']=lux_filt['values'].loc[(lux_filt['n_sensor']==7)].reset_index(drop=True)
datasetdir['outT']=outT_filt['values']#filtered_temperatures['values'] #NAN

# Check if there are any NaN values in the entire DataFrame
# has_nan = datasetOS['CO2_OS'].isna().any().any()

#resample of RH, lux, CO2 and outT
#CO2
new_indices_CO2 = np.linspace(0, len(co2_filt) - 1, len(datasetdir['indoorT']))
old_indices_CO2 = np.arange(len(co2_filt))
f1 = interpolate.interp1d(old_indices_CO2, co2_filt['values'], kind='linear')
CO2_res = f1(new_indices_CO2)
nan_indicesc = np.isnan(np.sum(CO2_res))

#outT
new_indices_outT = np.linspace(0, len(outT_filt) - 1, len(datasetdir['indoorT']))
old_indices_outT = np.arange(len(outT_filt))
f3 = interpolate.interp1d(old_indices_outT,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res = f3(new_indices_outT)
nan_indicesOR = np.isnan(np.sum(outT_res))

#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor=indoorT['n_sensor'].loc[(indoorT['n_sensor']==17)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor=sensor.reset_index(drop=True)
date=indoorT['DATE'].loc[(indoorT['n_sensor']==17)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date=date.reset_index(drop=True)  
date=pd.to_datetime(date, format = "%Y-%m-%d %H:%M:%S")

#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot=sPMV_calculation(datasetdir['indoorT'], datasetdir['indoorRH'],sensor,date) # 
spmv_tot=spmv_tot.reset_index(drop=True)

#create a dataset with the features we will include in the input dataframe
Xdua=pd.DataFrame()
Xdua['date']=datasetdir['date']
Xdua['indoor T']=datasetdir['indoorT']
Xdua['indoor RH']=datasetdir['indoorRH']
Xdua['CO2']=CO2_res
Xdua['outdoorT']=outT_res
Xdua['lux']=datasetdir['lux']
Xdua['sPMV']=spmv_tot['sPMV']
Xdua['room']='direzione_uff_a'

####ufficio qualità B
lux_filt['n_sensor']=lux_filt['n_sensor'].astype(float)
datasetUQ_B=pd.DataFrame()
datasetUQ_B['date_UQ_B']=indoorT['DATE'].loc[(indoorT['n_sensor']==16)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetUQ_B['indoorT_UQ_B']=indoorT['values'].loc[(indoorT['n_sensor']==16)].reset_index(drop=True)
datasetUQ_B['indoorRH_UQ_B']=indoorRH['values'].loc[(indoorRH['n_sensor']==16)].reset_index(drop=True)
datasetUQ_B['CO2_UQ_B']=co2_filt['values'].loc[(co2_filt['n_sensor']==39)].reset_index(drop=True) #NAN
datasetUQ_B['lux_UQ_B']=lux_filt['values'].loc[(lux_filt['n_sensor']==7)].reset_index(drop=True)
datasetUQ_B['outT']=outT_filt['values']#filtered_temperatures['values'] #NAN
#resample of RH, lux, CO2 and outT
#CO2
new_indices_CO2_UQ_B = np.linspace(0, len(co2_filt) - 1, len(datasetUQ_B['indoorT_UQ_B']))
old_indices_CO2_UQ_B = np.arange(len(co2_filt))
f1_UQ_B = interpolate.interp1d(old_indices_CO2_UQ_B, co2_filt['values'], kind='linear')
CO2_res_UQ_B = f1_UQ_B(new_indices_CO2_UQ_B)
nan_indicesc_UQ_B = np.isnan(np.sum(CO2_res_UQ_B))

#outT
new_indices_outT_UQ_B = np.linspace(0, len(outT_filt) - 1, len(datasetUQ_B['indoorT_UQ_B']))
old_indices_outT_UQ_B = np.arange(len(outT_filt))
f3_UQ_B = interpolate.interp1d(old_indices_outT_UQ_B,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res_UQ_B = f3_UQ_B(new_indices_outT_UQ_B)
nan_indicesOR_UQ_B = np.isnan(np.sum(outT_res_UQ_B))

#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor_UQ_B=indoorT['n_sensor'].loc[(indoorT['n_sensor']==16)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor_UQ_B=sensor_UQ_B.reset_index(drop=True)
date_UQ_B=datasetUQ_B['date_UQ_B'].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date_UQ_B=date_UQ_B.reset_index(drop=True)  
date_UQ_B=pd.to_datetime(date_UQ_B, format = "%Y-%m-%d %H:%M:%S")

#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot_UQ_B=sPMV_calculation(datasetUQ_B['indoorT_UQ_B'], datasetUQ_B['indoorRH_UQ_B'],sensor_UQ_B,date_UQ_B) # 
spmv_tot_UQ_B=spmv_tot_UQ_B.reset_index(drop=True)

#create a dataset with the features we will include in the input dataframe
X_UQ_B=pd.DataFrame()
X_UQ_B['date']=datasetUQ_B['date_UQ_B']
X_UQ_B['indoor T']=datasetUQ_B['indoorT_UQ_B']
X_UQ_B['indoor RH']=datasetUQ_B['indoorRH_UQ_B']
X_UQ_B['CO2']=CO2_res_UQ_B
X_UQ_B['outdoorT']=outT_res_UQ_B
X_UQ_B['lux']=datasetUQ_B['lux_UQ_B']
X_UQ_B['sPMV']=spmv_tot_UQ_B['sPMV']
X_UQ_B['room']='ufficio_q_B'


####Sala riunioni 1
lux_filt['n_sensor']=lux_filt['n_sensor'].astype(float)
datasetSR1=pd.DataFrame()
datasetSR1['date']=indoorT['DATE'].loc[(indoorT['n_sensor']==10)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
datasetSR1['indoorTSR1']=indoorT['values'].loc[(indoorT['n_sensor']==10)].reset_index(drop=True)
datasetSR1['indoorSR1']=indoorRH['values'].loc[(indoorRH['n_sensor']==10)].reset_index(drop=True)
datasetSR1['CO2_SR1']=co2_filt['values'].loc[(co2_filt['n_sensor']==34)].reset_index(drop=True) #NAN
datasetSR1['lux_SR1']=lux_filt['values'].loc[(lux_filt['n_sensor']==8)].reset_index(drop=True)
datasetSR1['outT']=outT_filt['values']

#resample of RH, lux, CO2 and outT
#CO2
new_indices_CO2_SR1 = np.linspace(0, len(co2_filt) - 1, len(datasetSR1['indoorTSR1']))
old_indices_CO2_SR1 = np.arange(len(co2_filt))
f1_SR1 = interpolate.interp1d(old_indices_CO2_SR1, co2_filt['values'], kind='linear')
CO2_res_SR1= f1_SR1(new_indices_CO2_SR1)
nan_indicesc_SR1 = np.isnan(np.sum(CO2_res_SR1))

#outT
new_indices_outT_SR1 = np.linspace(0, len(outT_filt) - 1, len(datasetSR1['indoorTSR1']))
old_indices_outT_SR1 = np.arange(len(outT_filt))
f3_SR1 = interpolate.interp1d(old_indices_outT_SR1,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res_SR1 = f3_SR1(new_indices_outT_SR1)
nan_indicesOR_SR1 = np.isnan(np.sum(outT_res_SR1))

#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor_SR1=indoorT['n_sensor'].loc[(indoorT['n_sensor']==10)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor_SR1=sensor_UQ_B.reset_index(drop=True)
date_SR1=indoorT['DATE'].loc[(indoorT['n_sensor']==10)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date_SR1=date_UQ_B.reset_index(drop=True)  
date_SR1=pd.to_datetime(date_UQ_B, format = "%Y-%m-%d %H:%M:%S")

#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot_SR1=sPMV_calculation(datasetSR1['indoorTSR1'], datasetSR1['indoorSR1'],sensor_SR1,date_SR1) # 
spmv_tot_SR1=spmv_tot_SR1.reset_index(drop=True)

#create a dataset with the features we will include in the input dataframe
X_SR1=pd.DataFrame()
X_SR1['date']=datasetSR1['date']
X_SR1['indoor T']=datasetSR1['indoorTSR1']
X_SR1['indoor RH']=datasetSR1['indoorSR1']
X_SR1['CO2']=CO2_res_SR1
X_SR1['outdoorT']=outT_res_SR1
X_SR1['lux']=datasetSR1['lux_SR1']
X_SR1['sPMV']=spmv_tot_SR1['sPMV']
X_SR1['room']='riunioni1'

####Segreteria
lux_filt['n_sensor']=lux_filt['n_sensor'].astype(float)
dataset_segr=pd.DataFrame()
dataset_segr['date_segr']=indoorT['DATE'].loc[(indoorT['n_sensor']==10)].reset_index(drop=True)#.loc[(indoorT['n_sensor']==6)].reset_index(drop=True)
dataset_segr['indoorT_segr']=indoorT['values'].loc[(indoorT['n_sensor']==10)].reset_index(drop=True)
dataset_segr['indoorRH_segr']=indoorRH['values'].loc[(indoorRH['n_sensor']==10)].reset_index(drop=True)
dataset_segr['CO2_segr']=co2_filt['values'].loc[(co2_filt['n_sensor']==34)].reset_index(drop=True) #NAN
dataset_segr['lux_segr']=lux_filt['values'].loc[(lux_filt['n_sensor']==8)].reset_index(drop=True)
dataset_segr['outT']=outT_filt['values']

#resample of RH, lux, CO2 and outT
#CO2
new_indices_CO2_segr = np.linspace(0, len(co2_filt) - 1, len(dataset_segr['indoorT_segr']))
old_indices_CO2_segr = np.arange(len(co2_filt))
f1_segr = interpolate.interp1d(old_indices_CO2_segr, co2_filt['values'], kind='linear')
CO2_res_segr= f1_segr(new_indices_CO2_segr)
nan_indicesc_segr = np.isnan(np.sum(CO2_res_segr))

#outT
new_indices_outT_segr = np.linspace(0, len(outT_filt) - 1, len(dataset_segr['indoorT_segr']))
old_indices_outT_segr = np.arange(len(outT_filt))
f3_segr = interpolate.interp1d(old_indices_outT_segr,outT_filt['values'], kind='linear', fill_value='extrapolate')
outT_res_segr = f3_segr(new_indices_outT_segr)
nan_indicesOR_segr = np.isnan(np.sum(outT_res_segr))

#calcolo sPMV
#select from the overall dataset the ID of sensors and the date in correspondance of data_type == 'indoor T' or  data_type == 'indoor RH'
sensor_segr=indoorT['n_sensor'].loc[(indoorT['n_sensor']==10)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
sensor_segr=sensor_segr.reset_index(drop=True)
date_segr=indoorT['DATE'].loc[(indoorT['n_sensor']==10)].reset_index(drop=True) #.loc[indoorT['values']=='indoor T']
date_segr=date_segr.reset_index(drop=True)  
date_segr=pd.to_datetime(date_segr, format = "%Y-%m-%d %H:%M:%S")

#sPMV calculation for all the year (from month 1 to month 12)
spmv_tot_segr=sPMV_calculation(dataset_segr['indoorT_segr'], dataset_segr['indoorRH_segr'],sensor_segr,date_segr) # 
spmv_tot_segr=spmv_tot_segr.reset_index(drop=True)

#create a dataset with the features we will include in the input dataframe
X_segr=pd.DataFrame()
X_segr['date']=dataset_segr['date_segr']
X_segr['indoor T']=dataset_segr['indoorT_segr']
X_segr['indoor RH']=dataset_segr['indoorRH_segr']
X_segr['CO2']=CO2_res_segr
X_segr['outdoorT']=outT_res_segr
X_segr['lux']=dataset_segr['lux_segr']
X_segr['sPMV']=spmv_tot_segr['sPMV']
X_segr['room']='riunioni'

####concat the dataframes 
result = pd.concat([X, X_SR1, X_SR3, X_UA_A, Xdua,X_UA_B,X_UQ_A,X_SR2, X_UQ_B, X_segr], axis=0, ignore_index=True)
result2=result.sort_values(by=['date'])
####save the total dataset
#result.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/input dataset/focchi_input_dataset_61223_allrooms.csv')

####check the correlation between the input variables to select the valuable features
import seaborn as sns
correlation_matrix = result.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix ')
plt.show()