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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf

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
    dataset['date']=pd.to_datetime(dataset['date'],  format = "%Y-%m-%d %H:%M:%S")
    dataset['hour']=dataset['date'].dt.hour
    dataset['day_of_week']=dataset['date'].dt.weekday #weekday mi da il numero intero riferito al giorno, altrimenti con la stringa mi dava errore
    if 'Unnamed: 0' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 0'])
    dataset=dataset.reset_index(drop=True)
    ####add outdoor for the next day
    outNext=dataset['outT'].shift(-24)
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

def dataset_input1(dataset):
    #add hour and day of the week 
    dataset['hour']=dataset['date'].dt.hour
    dataset['day_of_week']=dataset['date'].dt.weekday #weekday mi da il numero intero riferito al giorno, altrimenti con la stringa mi dava errore
    if 'Unnamed: 0' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 0'])
    dataset=dataset.reset_index(drop=True)
    # ####add outdoor for the next day
    # outNext=dataset['outdoorT'].shift(-24)
    # dataset['outNext']=outNext 
    #dividere input dataset giorno per giorno 
    # dataset_crop=dataset[0:len(dataset)-24] 
    dataset['day_of_week_sin'] = np.sin(2 * np.pi * dataset['day_of_week']/7)
    dataset['day_of_week_cos'] = np.cos(2 * np.pi * dataset['day_of_week']/7)
    dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour']/24)
    dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour']/24)
    #add columns
    # dataset['day_of_week_sin']=dataset['day_of_week_sin']
    # dataset['day_of_week_cos']=dataset['day_of_week_cos']
    # dataset['hour_sin']=dataset['hour_sin']
    # dataset['hour_cos']=dataset['hour_cos']
    return dataset

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


#target = deve essere una stringa
# def lstm_test(dataset, target, url_model, percentage_train_test, sequence_length_value, forecast_horizon_value):
    
#     if target=='Tin': 
#         #salva come  focchi
#         Xt = dataset[[ 'indoor T', 'outdoorT', 'outNext', 'day_of_week_sin','day_of_week_cos','hour_sin', 'hour_cos']] 
#         yt = dataset['indoor T']
#         ####split the features dataset
#         train_size = int(len(Xt) * percentage_train_test)  # 80% for training, 20% for testing
#         X_train, X_test = Xt[:train_size], Xt[train_size:]
#         y_train, y_test = yt[:train_size], yt[train_size:]

#         ####Normalize the training data and test data
#         scaler = MinMaxScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         #application of the above function
#         X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length_value, forecast_horizon_value)
#         X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length_value, forecast_horizon_value)
        
#         ####load of trained LSTM network
#         loaded_modelT = tf.keras.models.load_model(url_model)
#         # val = X_test_seq1[1,:,:]
#         y_pred = loaded_modelT.predict(X_test_seq)
#         mse = mean_squared_error(y_test_seq, y_pred)
#         mae = mean_absolute_error(y_test_seq, y_pred)
#         print(f' MSE: {mse}, MAE: {mae}') 
        
#         return y_pred, y_test_seq, y_test, mae, mse
    
#     elif target =='RH':
#             #salva come  focchi
#             Xrh = dataset[[ 'indoor RH', 'outdoorT', 'outNext', 'day_of_week_sin','day_of_week_cos','hour_sin', 'hour_cos'  ]] #'day_of_week', 'hour',,'outdoorT', 'lux', 'indoor RH', 'CO2', 'indoor T',
#             yrh = dataset['indoor RH']

#             ####split the features dataset
#             train_size = int(len(Xrh) * percentage_train_test)  # 80% for training, 20% for testing
#             X_train, X_test = Xrh[:train_size], Xrh[train_size:]
#             y_train, y_test = yrh[:train_size], yrh[train_size:]

#             ####Normalize the training data and test data
#             scaler = MinMaxScaler()
#             X_train_scaled = scaler.fit_transform(X_train)
#             X_test_scaled = scaler.transform(X_test)

#             #application of the above function
#             X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length_value, forecast_horizon_value)
#             X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length_value, forecast_horizon_value)
    
#             ####load of trained LSTM network
#             loaded_modelrh = tf.keras.models.load_model(url_model)
#             # val = X_test_seq1[1,:,:]
#             y_pred = loaded_modelrh.predict(X_test_seq)
#             mse = mean_squared_error(y_test_seq, y_pred)
#             mae = mean_absolute_error(y_test_seq, y_pred)
#             print(f' MSE: {mse}, MAE: {mae}') 
            
#             return y_pred, y_test_seq, y_test, mae, mse 
    
 
# def lstm_test(X, y, url_model, percentage_train_test, sequence_length_value, forecast_horizon_value):
#     train_size = int(len(X) * percentage_train_test)  # 80% for training, 20% for testing
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]

#     ####Normalize the training data and test data
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     ### Create sequences for LSTM model
#     sequence_length = sequence_length_value  # 1 week (24h * 7)
#     forecast_horizon = forecast_horizon_value  # Predicting the next  hour

#     X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length, forecast_horizon)
#     X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length, forecast_horizon)

#     ####load of trained LSTM network
#     loaded_modelT = tf.keras.models.load_model(url_model)
#     # val = X_test_seq1[1,:,:]
#     y_pred = loaded_modelT.predict(X_test_seq)
    
#     return y_pred, y_test_seq, y_test 

# def TSV_test(X, y, url_model, percentage_train_test, sequence_length_value, forecast_horizon_value):
#     train_size = int(len(X) * percentage_train_test)  # 80% for training, 20% for testing
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
#     ####Normalize the training data and test data
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     ### Create sequences for LSTM model
#     sequence_length = sequence_length_value  # 1 week (24h * 7)
#     forecast_horizon = forecast_horizon_value  # Predicting the next  hour
#     X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length, forecast_horizon)
#     X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length, forecast_horizon)
#     ####load of trained LSTM network
#     loaded_modelT = tf.keras.models.load_model(url_model)
#     # val = X_test_seq1[1,:,:]
#     y_pred = loaded_modelT.predict(X_test_seq)
#     return y_pred, y_test_seq, y_test
    

# def TSV_train(X, y, percentage_train_test, sequence_length_value, forecast_horizon_value):
#     train_size = int(len(X) * percentage_train_test)  # 80% for training, 20% for testing
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
#     ####Normalize the training data and test data
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     ### Create sequences for LSTM model
#     sequence_length = sequence_length_value  # 1 week (24h * 7)
#     forecast_horizon = forecast_horizon_value  # Predicting the next  hour
#     X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length, forecast_horizon)
#     X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length, forecast_horizon)
#     ####load of trained LSTM network
#     model = Sequential() #per salvare modello traianto .save di sequential (es model.model.save())
#     model.add(LSTM(64,return_sequences=True, input_shape=(sequence_length, X_train.shape[1])))
#     model.add(Dropout(0.2))
#     model.add(LSTM(64))
#     model.add(Dropout(0.2))
#     model.add(Dense(forecast_horizon))  
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'], run_eagerly=True)
#     #EarlyStopping
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min')
#     #Train 
#     history = model.fit(X_train_seq, y_train_seq, epochs=15, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stopping])
#     y_pred = model.predict(X_test_seq)
#     return  y_pred

# def zscore(a, window, thresh=2, return_all=False):
#     roll = a.rolling(window=window, min_periods=1, center=True)
#     avg = roll.mean()
#     std = roll.std(ddof=0)
#     z = a.sub(avg).div(std)   
#     m = z.between(-thresh, thresh)
    
#     if return_all:
#         return z, avg, std, m
#     return a.where(m, avg)

# percentage_train_test=0.8
# sequence_length_value=168
# forecast_horizon_value=1
# n_neur=64
# n_epoc=15
# batch=32
# def Train_lstm_for_tin(X, y, percentage_train_test, sequence_length_value, forecast_horizon_value, n_neur, n_epoc, batch):
#     ####split the features dataset
#     train_size = int(len(X) * percentage_train_test)  # 80% for training, 20% for testing
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
#     ####Normalize the training data and test data
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     ### Create sequences for LSTM model
#     sequence_length = sequence_length_value  # 1 week (24h * 7)
#     forecast_horizon = forecast_horizon_value  # Predicting the next hour

#     #application of the above function
#     X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length, forecast_horizon)
#     X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length, forecast_horizon)

#     ####LSTM
#     model = Sequential() #per salvare modello traianto .save di sequential (es model.model.save())
#     model.add(LSTM(n_neur,return_sequences=True, input_shape=(sequence_length, X_train.shape[1])))
#     model.add(Dropout(0.2))
#     model.add(LSTM(n_neur))
#     model.add(Dropout(0.2))
#     model.add(Dense(forecast_horizon))  
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'], run_eagerly=True)
#     #EarlyStopping
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min')
#     #Train 
#     history = model.fit(X_train_seq, y_train_seq, epochs=n_epoc, batch_size=batch, validation_split=0.1, verbose=1, callbacks=[early_stopping])
#     model.save('lstm_Tin_'+str(forecast_horizon_value)+'__'+str(n_neur)+'__'+str(date.today())+'.h5')
    
#     y_pred = model.predict(X_test_seq)
#     mse = mean_squared_error(y_test_seq, y_pred)
#     mae = mean_absolute_error(y_test_seq, y_pred)
#     print(f' MSE: {mse}, MAE: {mae}') 
    
#     return {'predicted_value_Tin': y_pred, 
#             'actual_value_tin':y_test_seq}
    
# def Train_lstm_for_RH(X, y, percentage_train_test, sequence_length_value, forecast_horizon_value, n_neur, n_epoc, batch):
#     ####split the features dataset
#     train_size = int(len(X) * percentage_train_test)  # 80% for training, 20% for testing
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
#     ####Normalize the training data and test data
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     ### Create sequences for LSTM model
#     sequence_length = sequence_length_value  # 1 week (24h * 7)
#     forecast_horizon = forecast_horizon_value  # Predicting the next hour

#     #application of the above function
#     X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length, forecast_horizon)
#     X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length, forecast_horizon)

#     ####LSTM
#     model = Sequential() #per salvare modello traianto .save di sequential (es model.model.save())
#     model.add(LSTM(n_neur,return_sequences=True, input_shape=(sequence_length, X_train.shape[1])))
#     model.add(Dropout(0.2))
#     model.add(LSTM(n_neur))
#     model.add(Dropout(0.2))
#     model.add(Dense(forecast_horizon))  
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'], run_eagerly=True)
#     #EarlyStopping
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min')
#     #Train 
#     history = model.fit(X_train_seq, y_train_seq, epochs=n_epoc, batch_size=batch, validation_split=0.1, verbose=1, callbacks=[early_stopping])
#     model.save('lstm_RH_'+str(forecast_horizon_value)+'__'+str(n_neur)+'__'+str(date.today())+'.h5')
    
#     y_pred = model.predict(X_test_seq)
#     mse = mean_squared_error(y_test_seq, y_pred)
#     mae = mean_absolute_error(y_test_seq, y_pred)
#     print(f' MSE: {mse}, MAE: {mae}') 
    
#     return y_pred  





####RICKY
def predict_with_lstm(model, input_data, scaler_y_TSV, scaler_y_payload):
    predictions = model.predict(input_data)
    y_pred_TSV = predictions[:, :forecast_horizon]
    y_pred_payload = predictions[:, forecast_horizon:]
    y_pred_TSV_inv = scaler_y_TSV.inverse_transform(y_pred_TSV)
    y_pred_payload_inv = scaler_y_payload.inverse_transform(y_pred_payload)
    return y_pred_TSV_inv, y_pred_payload_inv

# Funzione obiettivo per ottimizzare la temperatura di setpoint
def objective_function(temp, model, scaler_y_TSV, scaler_y_payload, target_TSV_range, sequence, num_features):
    temp_sequence = sequence.copy()
    temp_sequence[:, 0] = temp
    temp_sequence = temp_sequence.reshape(1, sequence_length, num_features)
    
    y_pred_TSV, y_pred_payload = predict_with_lstm(model, temp_sequence, scaler_y_TSV, scaler_y_payload)
    tsv_pred = y_pred_TSV[0, 0]
    
    # Print per debug
    print(f"Temp: {temp}, TSV Predetto: {tsv_pred}, Energia Predetta: {y_pred_payload[0, 0]}")
    
    tsv_penalty = 0
    if tsv_pred < target_TSV_range[0]:
        tsv_penalty = (target_TSV_range[0] - tsv_pred) ** 2
    elif tsv_pred > target_TSV_range[1]:
        tsv_penalty = (tsv_pred - target_TSV_range[1]) ** 2
    
    return y_pred_payload[0, 0] + tsv_penalty

# Funzione per trovare il setpoint ideale tra le temperature di test
def find_optimal_setpoint_from_tests(model, scaler_y_TSV, scaler_y_payload, target_TSV_range, sequence, num_features, test_temperatures):
    best_temp = None
    best_cost = float('inf')
    
    for temp in test_temperatures:
        temp_sequence = sequence.copy()
        temp_sequence[:, 0] = temp
        temp_sequence = temp_sequence.reshape(1, sequence_length, num_features)
        
        y_pred_TSV, y_pred_payload = predict_with_lstm(model, temp_sequence, scaler_y_TSV, scaler_y_payload)
        
        tsv_penalty = 0
        if y_pred_TSV[0, 0] < target_TSV_range[0]:
            tsv_penalty = 10*(target_TSV_range[0] - y_pred_TSV[0, 0]) ** 2
        elif y_pred_TSV[0, 0] > target_TSV_range[1]:
            tsv_penalty = 10*(y_pred_TSV[0, 0] - target_TSV_range[1]) ** 2
        
        regularization_penalty = (temp - 22) ** 2
        cost = y_pred_payload[0, 0] + tsv_penalty + regularization_penalty
        
        # Print per debug
        print(f"Testing Temp: {temp}, Cost: {cost}")
        
        if cost < best_cost:
            best_cost = cost
            best_temp = temp
    
    return best_temp

# Predici TSV e consumo energetico per le finestre di test
def predict_and_optimize_setpoints(model, X_test, scaler_y_TSV, scaler_y_payload):
    results = []
    test_temperatures = np.linspace(20, 26, 14)  # Definizione dei valori di test per la temperatura

    for i in range(len(X_test)):
        sequence = X_test[i].reshape(sequence_length, X_sequences.shape[2])
        y_pred_TSV, y_pred_payload = predict_with_lstm(model, X_test[i].reshape(1, sequence_length, X_sequences.shape[2]), scaler_y_TSV, scaler_y_payload)
        
        # Trova il setpoint ideale per la finestra corrente
        optimal_temp = find_optimal_setpoint_from_tests(model, scaler_y_TSV, scaler_y_payload, (-0.5, 0.5), sequence, X_sequences.shape[2], test_temperatures)
        
        # Predici TSV ed energia con il setpoint ottimale
        optimal_sequence = sequence.copy()
        optimal_sequence[:, 0] = optimal_temp  # Modifica solo la temperatura
        optimal_sequence = optimal_sequence.reshape(1, sequence_length, X_sequences.shape[2])
        y_new_TSV, y_new_payload = predict_with_lstm(model, optimal_sequence, scaler_y_TSV, scaler_y_payload)
        
        # Aggiungi i risultati per la finestra corrente
        results.append({
            'Finestra': i + 1,
            'Setpoint Ideale': optimal_temp,
            'Temperatura di Test': np.nan,  # Placeholder per la temperatura di test ideale
            'TSV Predetto (Iniziale)': y_pred_TSV[0, 0],
            'Energia Predetta (Iniziale)': y_pred_payload[0, 0],
            'TSV con Nuovo Setpoint': y_new_TSV[0, 0],
            'Energia con Nuovo Setpoint': y_new_payload[0, 0]
        })
    
    results_df = pd.DataFrame(results)
    
    # Aggiorna la colonna 'Temperatura di Test' con la temperatura ideale testata
    for i in range(len(results_df)):
        sequence = X_test[i].reshape(sequence_length, X_sequences.shape[2])
        best_temp = find_optimal_setpoint_from_tests(
            model, scaler_y_TSV, scaler_y_payload, 
            (-0.5, 0.5), sequence, X_sequences.shape[2], 
            np.linspace(20, 26, 14)
        )
        results_df.at[i, 'Temperatura di Test'] = best_temp
    
    return results_df    


# plt.figure()
# plt.plot(dataset_crop['date'], dataset_crop['outdoorT'], c='red')
# plt.plot(dataset_crop['date'], dataset_crop['indoor T'], c='blue')

# a=dataset_crop['room'].unique()
# for i in range(len(a)):
#     #plot env data
#     fig = plt.figure(figsize=(20, 10))
#     gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
#     ax1 = fig.add_subplot(gs[0])
#     ax1.plot(dataset_crop['date'].loc[(dataset_crop['room']==a[i])], dataset_crop['indoor T'].loc[(dataset_crop['room']==a[i])], label='indoor T (°C)', color='blue')
#     ax1.tick_params(axis='x', rotation=45)
#     ax1.set_ylabel('Temperature (°C)', fontsize=18)
#     ax2 = ax1.twinx()
#     ax2.plot( dataset_crop['date'].loc[dataset_crop['room']==a[i]], dataset_crop['indoor RH'].loc[dataset_crop['room']==a[i]], label='indoor RH (%)', color='red')
#     ax2.set_ylabel('indoor RH (%)', c='r', fontsize=18)
#     ax2.tick_params('y', colors='r')
#     ax1.set_title('Trends of the environmental factors -'+ str(a[i]) , fontsize=18)
#     ax2 = fig.add_subplot(gs[1], sharex=ax1)
#     ax2.plot(dataset_crop['date'].loc[dataset_crop['room']==a[i]], dataset_crop['outdoorT'].loc[dataset_crop['room']==a[i]], label='out T', color='green')
#     ax2.set_ylabel('out T', fontsize=18)
#     ax2.tick_params(axis='x', rotation=45)
#     plt.tight_layout()
#     plt.show()
