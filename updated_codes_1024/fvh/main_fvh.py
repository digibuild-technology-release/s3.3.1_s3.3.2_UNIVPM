# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:18:38 2024

@author: 39324
"""

import pandas as pd
import numpy as np
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
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from processing_functions import *
from tensorflow.keras.models import load_model
from datetime import timedelta
from sPMV_v1 import *


####Request for input data from the datalake: indoor T, Outdoor T, Outdoor of the next day, Datetime
####create the dataframe
# df = pd.DataFrame({
#     'date': datetime, #replace based on how the retrieved data are defined
#     'indoorT': df_temp['value'], #replace based on how the retrieved data are defined
#     'indoorRH': df_rh['value'], #replace based on how the retrieved data are defined
#     'outdoorT': df_out['value'], #replace based on how the retrieved data are defined
#     'outNext': df_out_next['value'] #replace based on how the retrieved data are defined
# })


#%%%% PREDICTIONS of indoor T and indoor RH
df=pd.read_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/Attivita_Vittoria/PAPERS/paper_sPMV/python/2_case_study_FVH/dati_stanze_FVH/dataset_test_singleroom117.csv')

df['date']=pd.to_datetime(df['DATE'])

#resample data in hourly data
df.set_index('date', inplace=True)  # Set the timestamp column as index
df = df.resample('H').mean().interpolate()
df=df.reset_index()

####process the data to create inPut for LSTM
input_datasetor=dataset_input(df)

#select 1 week of hist. data
input_dataset=input_datasetor[2:2+168].reset_index(drop=True) #seleziono la prima settimana (parto dal 2 semplicemente per far partire la settimana dalle 00:00)

####------------------------------------------indoor T---------------------------------------------------------------------
#select column of interest
Xt = input_dataset[[ 'indoor T', 'outT', 'outNext', 'day_of_week_sin','day_of_week_cos','hour_sin', 'hour_cos'  ]]#.round(2)
####Normalize the training data and test data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(Xt)

sequence_length=168
forecast_horizon=24

####Reshape the data for LSTM
X_seqt=scaled_data.reshape(1, 168, 7)

####load the trained model for indoor T
loaded_model1=load_model('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/fvh/modelli/tin_pred_fvh_191124.h5') #replace the url with the correct one

####create sequences for the LSTM network (use one week to predict the next 24 hours) -- Indoor T
y_predT=loaded_model1.predict(X_seqt)

####denormalized prediction - T in
yPredDt= np.round(denorm(y_predT, Xt, scaler).reshape(24,),1)

####------------------------------------------indoor RH---------------------------------------------------------------------
#select column of interest
Xrh = input_dataset[[ 'indoor RH', 'outT', 'outNext', 'day_of_week_sin','day_of_week_cos','hour_sin', 'hour_cos'  ]]
####Normalize the training data and test data
scaled_datarh = scaler.fit_transform(Xrh)
####create sequences for the LSTM network (use one week to predict the next 24 hours) -- Indoor RH
X_seqrh=scaled_datarh.reshape(1, 168, 7)
####load the trained model for indoor RH
loaded_model2=load_model('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/Attivita_Vittoria/PAPERS/paper_sPMV/python/2_case_study_FVH/rh_pred_fvh_201124.h5')

# loaded_model2=load_model('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_final/Tin_RH_pred/saved models/lstm_RH_24_64_add_810024_oneroom_norm2.h5')
####make the prediction of indoor RH
y_predRH=loaded_model2.predict(X_seqrh)
####denormalize predicted values - RH
yPredDrh= np.round(denorm(y_predRH, Xrh, scaler).reshape(24,), 1)

####sPMV calculation - versione 1
#create the dataset to calculate sPMV of the next 24 hours 
inputcomfortcalc = pd.DataFrame({
    'date': timestamps, 
    'pred_indoorT': yPredDt.reshape(24,), 
    'pred_indoorRH': yPredDrh.reshape(24,), 
})

####SPMV +24 HOURS
spmv_pred=sPMV_calculation(inputcomfortcalc['pred_indoorT'], inputcomfortcalc['pred_indoorRH'], inputcomfortcalc['date'] ) #inputcomfortcalc['sensor'], 
spmv_pred24=spmv_pred['sPMV']
mean_spmv24=np.mean(spmv_pred24).round(1)

#%%% TSV - FVH
from sklearn.ensemble import RandomForestRegressor
import pickle
tsv_model = pickle.load(open('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/fvh/modelli/RF_tsv.pkl', 'rb')) #'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/models/random_forest.pkl'
X_now = pd.DataFrame()
X_now['indoor T_y'] = input_dataset['indoor T']
X_now['indoor RH_y'] = input_dataset['indoor RH']
X_now['outT'] = input_dataset['outT']

tsv_now = tsv_model.predict(X_now)
tsv_now=np.round(tsv_now, 2)
mean_tsv_now=np.mean(tsv_now).round(1)


####TSV +24 HOURS
X24 = pd.DataFrame()
#link to API for outdoor of the next day
t=timestamps[0]
ind = input_datasetor[input_datasetor['date'] == t]
i = ind.index
perout = input_datasetor['outNext'].iloc[i.values[0]: (i.values[0])+24]

X24['indoor T_y'] = yPredDt.reshape(24,)
X24['indoor RH_y'] = yPredDrh.reshape(24,)
X24['outT'] = perout

tsv_tomorrow = tsv_model.predict(X24)
mean_tsv_tomorrow=np.mean(tsv_tomorrow).round(0)







