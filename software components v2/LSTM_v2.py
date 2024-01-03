# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:03:06 2023

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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import mlflow
import time
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error

####import your dataset
dataset_dir='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/input dataset/focchi_input_dataset_201123.csv'
dataset=pd.read_csv(dataset_dir, sep=',')
dataset['date']=pd.to_datetime(dataset['date'],  format = "%Y-%m-%d %H:%M:%S")
dataset=dataset.drop(['Unnamed: 0'], axis=1)
# dataset = dataset.sort_values(by='date', ascending=True)
dataset=dataset.reset_index(drop=True)

#### Extract the features and label from the imported dataset
X = dataset[['indoor T','indoor RH', 'CO2', 'outdoorT', 'lux']] 
y = dataset['sPMV']

# dataset_dir='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/edf_pilot/dataset_FFN.csv'
# dataset=pd.read_csv(dataset_dir, sep=',')
# dataset['date']=pd.to_datetime(dataset['DateTime'],  format = "%Y-%m-%d %H:%M:%S")
# dataset=dataset.drop(['DateTime'], axis=1)
# # dataset = dataset.sort_values(by='date', ascending=True)
# # dataset=dataset.reset_index(drop=True)

# #### Extract the features and label from the imported dataset
# X = dataset[['env_temp','env_rh', 'co2_co2', 'Temperature_out']] 
# y = dataset['sPMV']

####split the features dataset
train_size = int(len(X) * 0.8)  # 80% for training, 20% for testing
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

####Normalize the training data and test data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#### Create sequences for LSTM model
sequence_length = 168  # 1 week (24h * 7)
forecast_horizon = 1  # Predicting the next hour
# forecast_horizon = 24  # Predicting the next 24 hours

def create_sequences(input_data, output_data, sequence_length, forecast_horizon):
    X_sequences, y_sequences = [], []
    
    for i in range(len(input_data) - sequence_length - forecast_horizon + 1):
        X_sequences.append(input_data[i:i + sequence_length])
        y_sequences.append(output_data[i + sequence_length:i + sequence_length + forecast_horizon])
    return np.array(X_sequences), np.array(y_sequences)

#application of the above function
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length, forecast_horizon)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length, forecast_horizon)

####STM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])))
model.add(LSTM(64))
model.add(Dense(forecast_horizon))  
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
#EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min')
#Train 
history = model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stopping])

# Predicting using the test set
y_pred = model.predict(X_test_seq)

# Align the dates (taking the date column from the original dataset) of the test set with the sequences
test_dates = dataset.loc[y_test.index]
test_dates = test_dates.reset_index(drop=True)

# Adjust the date index to align with the sequences
adjusted_test_dates = test_dates[sequence_length + forecast_horizon - 1:].reset_index(drop=True)

# Select a specific period to visualize - for example, the first 24-hour period in the test set
period_to_visualize = 324# Change this to visualize different periods (autumn, winter,spring,summer)
# Calculate the start date for the selected period
start_date = adjusted_test_dates['date'].iloc[period_to_visualize]
print(start_date)

# Generate timestamps for the 24-hour period from the chosen start date
timestamps = pd.date_range(start=start_date, periods=len(y_test_seq), freq='H')

# Extract the actual and predicted values for this period
actual_values = y_test_seq[period_to_visualize:period_to_visualize+24] #uncomment if forecasting horizon is 1 hour
predicted_values = y_pred[period_to_visualize:period_to_visualize+24]

# actual_values = y_test_seq[period_to_visualize] #uncomment if forecasting horizon is 24 hour
# predicted_values = y_pred[period_to_visualize]

mse = mean_squared_error(y_test_seq, y_pred)
mae = mean_absolute_error(y_test_seq, y_pred)
# mape=mean_absolute_percentage_error(y_test_seq, y_pred)
print(f' MSE: {mse}, MAE: {mae}') # MAPE: {mape}

# Generate timestamps for the 24-hour period for x-axis
timestamps = pd.date_range(start=start_date, periods=24, freq='H')  # Adjust the start date as needed

# Plotting the actual vs predicted values. TAKE NOTE: provide 1 plot for each season (autumn, winter, spring, summer)
plt.figure(figsize=(12, 6))
plt.plot(timestamps,actual_values, label='Actual Values', marker='o')
plt.plot( timestamps,predicted_values, label='Predicted Values', marker='x')
plt.title('Comparison of Actual and Predicted sPMV Values Over 24 hour ')
plt.xlabel('Time')
plt.ylabel('sPMV')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#scatterplot between the test and the predicted value
plt.figure(figsize=(20,10))
plt.plot(y_test_seq)
plt.plot( y_pred)

plt.xlabel('true')
plt.ylabel('predicted')
plt.title('Scatter plot - predicted sPMV VS true sPMV')


#%%%%save the trained model(s) you choose as pickle
import joblib
joblib.dump(model, 'finalized_model_lstm_v2.joblib')


# #%%%%save the trained model(s) on MLflow digibuild repository
# import os
# os.environ["MLFLOW_TRACKING_URI"]= "http://digibuild.epu.ntua.gr:5000/"
# os.environ["AWS_SECRET_ACCESS_KEY"]= "minio123"
# os.environ["MLFLOW_S3_ENDPOINT_URL"]= "http://digibuild.epu.ntua.gr:9000/"
# os.environ["AWS_ACCESS_KEY_ID"]= "minio"
# # define model input features and signature
# input_schema = Schema([
#   ColSpec("float", "indoor T"),
#   ColSpec("float", "indoor RH"),
#   ColSpec("float", "CO2"),
#   ColSpec("float", "outdoor T"),
#   ColSpec("float", "lux")
# ])

# output_schema = Schema([ColSpec("long", "sPMV")])
# # signature consists of model input and output schemas
# signature = ModelSignature(inputs=input_schema, outputs=output_schema)
# #experiment creation
# experiment_name = "FOCCHI_s331_lstm"
# model_name = "LSTM_t2_univpm"
# mlflow.set_experiment(experiment_name)

# with mlflow.start_run():
    
#     # mlflow log model
#     mlflow.sklearn.log_model(model, model_name, signature=signature, registered_model_name=model_name)
#     # mlflow log params
#     mlflow.log_param("units", 64)
#     mlflow.log_param("recurrent_dropout", 0.1)
#     mlflow.log_param("n_epochs", 5)
#     mlflow.log_param("batch_size", 32)
#     mlflow.log_metric("mse", mean_squared_error(y_test_seq, y_pred))
#     mlflow.log_metric("mae", mean_absolute_error(y_test_seq, y_pred))
#     mlflow.log_artifact(pred_path)