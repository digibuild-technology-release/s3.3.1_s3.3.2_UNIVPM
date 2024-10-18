# -*- coding: utf-8 -*-
"""
MAIN CODE FOR UCL PILOT 
NO INDOOR RH DATA ARE INCLUDED 
THE CODE CAN BE USED TO:1) PREDICT INDOOR T,
2) MEASURE TSV AT PREDENT,
3) MEASURE TSV FOR THE NEXT 24 HOURS
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
# from sPMV_v1 import sPMV_calculation
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,  RepeatVector, TimeDistributed, Dropout
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import mlflow
import time
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error
import calendar
from processing_functions import *
from sPMV_v1 import *
from tensorflow.keras.models import load_model
from datetime import timedelta
import pickle

####Request for input data from the datalake: indoor T, Outdoor T, Outdoor of the next day, Datetime
####create the dataframe
# df = pd.DataFrame({
#     'date': datetime, #replace based on how the retrieved data are defined
#     'indoorT': df_temp['value'], #replace based on how the retrieved data are defined
#     'outdoorT': df_out['value'], #replace based on how the retrieved data are defined
#     'outNext': df_out_next['value'] #replace based on how the retrieved data are defined
# })


####historical data to test the code

dataset_dir='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/dataset_ucl.csv'
df=pd.read_csv(dataset_dir, sep=',')
df['date']=pd.to_datetime(df['date'],  format = "%Y-%m-%d %H:%M:%S")
df = df.drop(columns=['Unnamed: 0'])
####process the data to create inout for LSTM
input_datasetor=dataset_input(df)
#select 1 week of hist. data
input_dataset=input_datasetor[0:168].reset_index(drop=True)

####------------------------------------------indoor T---------------------------------------------------------------------
#select column of interest
Xt = input_dataset[[ 'indoor T', 'outdoorT', 'outNext', 'day_of_week_sin','day_of_week_cos','hour_sin', 'hour_cos'  ]]
####Normalize the training data and test data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(Xt)
# sequence_length=168
# forecast_horizon=24

####Reshape the data for LSTM
X_seqt=scaled_data.reshape(1, 168, 7)

####load the trained model for indoor T
# loaded_model1=load_model('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/models/lstm_Tin_24_64_add_71024_room_norm_new2.h5') #replace the url with the correct one
loaded_model1=load_model('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/models/lstm_Tin_24_64_91024_room.h5') #replace the url with the correct one

####create sequences for the LSTM network (use one week to predict the next 24 hours) -- Indoor T
y_predT=loaded_model1.predict(X_seqt)
####denormalized prediction - T in
yPredDt= denorm(y_predT, Xt, scaler)

timestamps = pd.date_range(start=input_dataset['date'][len(input_dataset)-1] + timedelta(hours = 1), periods=24, freq='H')

####----------------------------------THERMAL SENSATION VOTE---------------------------------------------------------------------------------------
####TSV NOW
tsv_model = pickle.load(open('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/models/rf_UCL.pkl', 'rb'))
X_now = pd.DataFrame()
X_now['outdoor T'] = input_dataset['outdoorT']
X_now['indoor T'] = input_dataset['indoor T']
tsv_now = tsv_model.predict(X_now)

####TSV +24 HOURS
X24 = pd.DataFrame()
#link to API for outdoor of the next day
t=timestamps[0]
ind = input_datasetor[input_datasetor['date'] == t]
i = ind.index
perout = input_datasetor['outNext'].iloc[i.values[0]: (i.values[0])+24]
X24['outdoor T'] = perout
X24['indoor T'] = yPredDt.reshape(24,)
tsv_tomorrow = tsv_model.predict(X24)
