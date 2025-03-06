# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:27:31 2024

@author: utente
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from datetime import datetime
import os
from scipy import signal
import math
# from sPMV_v1 import sPMV_calculation
import seaborn as sns
from scipy.interpolate import interp1d
import time
import calendar
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def data_prep_traintest(dataset):
    # dataset_dir=directory
    # datasetor=pd.read_csv(dataset_dir, sep=',')
    if 'Unnamed: 0' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 0'])
    # datasetor=datasetor.drop(['Unnamed: 0'], axis=1)
    dataset['date']=pd.to_datetime(dataset['date'],  format = "%Y-%m-%d %H:%M:%S")
    months=range(1,13)
    DATASET_TRAIN=[]
    DATASET_TEST=[]
    df_TEST=pd.DataFrame()
    df_TRAIN=pd.DataFrame()
    for i in range(len(months)): 
        df=dataset.loc[dataset['date'].dt.month==months[i]]
        if df.size==0:
            continue
        else: 
            dataset_train, dataset_test = train_test_split(df, test_size=0.20,random_state=42)
            #sort by date (cronol. order)
            dataset_train=dataset_train.sort_values(by='date')
            dataset_test=dataset_test.sort_values(by='date')
            DATASET_TRAIN.append(dataset_train)
            DATASET_TEST.append(dataset_test)
           
    df_TRAIN=pd.concat(DATASET_TRAIN)
    df_TRAIN=df_TRAIN.reset_index(drop=True)
    # df_TRAIN.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_final/dataset_prep/dataset_train.csv')
    df_TEST=pd.concat(DATASET_TEST)
    df_TEST=df_TEST.reset_index(drop=True)
    # df_TEST.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_final/dataset_prep/dataset_test.csv')
    return df_TRAIN, df_TEST

def dataset_input(dataset):
    #add hour and day of the week 
    dataset['hour']=dataset['date'].dt.hour
    dataset['day_of_week']=dataset['date'].dt.weekday #weekday mi da il numero intero riferito al giorno, altrimenti con la stringa mi dava errore
    if 'Unnamed: 0' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 0'])
    dataset=dataset.reset_index(drop=True)
    ####add outdoor for the next day
    outNext=dataset['outdoorT'].shift(-24)
    dataset['outNext']=outNext 
    #dividere input dataset giorno per giorno 
    dataset_crop=dataset[0:len(dataset)-24] 
    dataset_crop['day_of_week_sin'] = np.sin(2 * np.pi * dataset_crop['day_of_week']/7)
    dataset_crop['day_of_week_cos'] = np.cos(2 * np.pi * dataset_crop['day_of_week']/7)
    dataset_crop['hour_sin'] = np.sin(2 * np.pi * dataset_crop['hour']/24)
    dataset_crop['hour_cos'] = np.cos(2 * np.pi * dataset_crop['hour']/24)
    
    #add columns
    dataset_crop['day_of_week_sin']=dataset_crop['day_of_week_sin']
    dataset_crop['day_of_week_cos']=dataset_crop['day_of_week_cos']
    dataset_crop['hour_sin']=dataset_crop['hour_sin']
    dataset_crop['hour_cos']=dataset_crop['hour_cos']
    
    return dataset_crop

def create_sequences(input_data, output_data, sequence_length, forecast_horizon):
    X_sequences, y_sequences = [], []
    for i in range(len(input_data) - sequence_length - forecast_horizon + 1):
        X_sequences.append(input_data[i:i + sequence_length])
        y_sequences.append(output_data[i + sequence_length:i + sequence_length + forecast_horizon])
    return np.array(X_sequences), np.array(y_sequences)

def denorm(y, X, scaler):
    # scaler=MinMaxScaler()
    y_pred_reshaped = np.zeros((y.shape[0], y.shape[1], X.shape[1]))
    # Fill the first column with the predicted and true values respectively
    y_pred_reshaped[:, :, 0] = y
    # Denormalize the predictions and the actual test sequences
    denorm_values = scaler.inverse_transform(y_pred_reshaped.reshape(-1, X.shape[1]))[:, 0].reshape(y.shape)
    return denorm_values


