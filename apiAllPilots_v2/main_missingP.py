from fastapi import FastAPI, Depends, HTTPException, Header, File
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests, os
# from pydantic import BaseModel
import mlflow
from mlflow import pyfunc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from io import StringIO
from funzioni_db_fvh import *
from datasharing_functions import *
import datetime
from adaptive_function import adaptive_model
from functions_collect_data import *
from sPMV_v1 import *
load_dotenv()
app = FastAPI()
security = HTTPBearer()

#focchi
#ML credentials
# MLFLOW_TRACKING_URI= os.getenv("MLFLOW_BASE_URL")
# AWS_SECRET_ACCESS_KEY= os.getenv("S3_SECRET_KEY")
# MLFLOW_S3_ENDPOINT_URL= os.getenv("MLFLOW_S3_URL")
# AWS_ACCESS_KEY_ID= os.getenv("S3_ACCESS_KEY")

os.environ["MLFLOW_TRACKING_URI"]= os.getenv("MLFLOW_BASE_URL")
os.environ["AWS_SECRET_ACCESS_KEY"]= os.getenv("S3_SECRET_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"]= os.getenv("MLFLOW_S3_URL")
os.environ["AWS_ACCESS_KEY_ID"]=  os.getenv("S3_ACCESS_KEY")


#datasharing credentials
username = os.getenv("IDM_USERNAME") 
password = os.getenv("IDM_PASSWORD") 
token = gen_token(username, password)

@app.get("/")
async def root():
    return{"message":"hi"} 

####UCL
@app.post("/adaptive")
async def calculate_comfort():  
    #richiesta datasharing per dati
    pilotdb = "ucl_test"
    #sensors id for indoor T and outdoor T
    sensors_tempIN=[''] 
    sensors_tempOUT=[ '' ] 

    #tempIN
    df_tempIN=execute_query(token, pilotdb, sensors_tempIN, time_from=None, time_to=None)
    df_tempIN['datetime']=pd.to_datetime(df_tempIN['datetime'], format="%Y%m%d%H%M%S")
    df_tempIN=df_tempIN.sort_values(by='datetime')
    df_tempIN=df_tempIN.reset_index(drop=True)
    #tempOUT
    df_tempOUT=execute_query(token, pilotdb, sensors_tempOUT, time_from=None, time_to=None)
    df_tempOUT['datetime']=pd.to_datetime(df_tempOUT['datetime'], format="%Y%m%d%H%M%S")
    df_tempOUT=df_tempOUT.sort_values(by='datetime')
    df_tempOUT=df_tempOUT.reset_index(drop=True)
   
    df=pd.DataFrame()
    df['date']=df_tempIN['datetime']
    df['indoor T']=df_tempIN['value']
    df['outdoor T']=df_tempOUT['value']
    df['room']='xx'
    
    unique_rooms = df['room'].unique()
    results = {}
    # model_name= os.getenv("FOCCHI_CALCULATE_MODEL") #fill with ML uri with the saved model
    
    try:
    # Load the MLflow model
        # model = mlflow.sklearn.load_model(model_name)
        # print("Loaded model:", model)
        # print('model loaded')
        for room in unique_rooms:
                room_data = df[df['room'] == room][['indoor T', 'outdoor T', 'date']] #, 'outdoorT', 'lux'
                
                output_adapt=adaptive_model(room_df['indoor T'], room_data['outdoor T'], room_data['date'])
                # room_predictions = np.mean(predRoom)
                results[room] = output_adapt.tolist()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
    return{'comfort level (adaptive)': results}


####EDF
@app.post("/calculate_edf")
async def calculate_comfort():  
    #richiesta datasharing per dati
    pilotdb = "edf_test"
    #Temperature
    sensor_temp=['32852_0_84']
    #Humidity
    sensor_hum=['32840_0_72'] 
    #co2
    sensor_co2=['32835_0_67']
    
    #co2
    df_co2=execute_query(token, pilotdb, sensor_co2, time_from=None, time_to=None)
    df_co2['datetime']=pd.to_datetime(df_co2['datetime'], format="%Y%m%d%H%M%S")
    df_co2=df_co2.sort_values(by='datetime')
    df_co2=df_co2.reset_index(drop=True)
    # df_co2_res = np.interp(np.linspace(0, 1, len(df_temp)), np.linspace(0, 1, len(df_co2)), df_co2['value'])
    #temp
    df_temp=execute_query(token, pilotdb, sensor_temp, time_from=None, time_to=None)
    df_temp['datetime']=pd.to_datetime(df_temp['datetime'], format="%Y%m%d%H%M%S")
    df_temp=df_temp.sort_values(by='datetime')
    df_temp=df_temp.reset_index(drop=True)
    temps_res = np.interp(np.linspace(0, 1, len(df_co2)), np.linspace(0, 1, len(df_temp)), df_temp['value'])
    #rh
    df_rh=execute_query(token, pilotdb, sensor_hum, time_from=None, time_to=None)
    df_rh['datetime']=pd.to_datetime(df_rh['datetime'], format="%Y%m%d%H%M%S")
    df_rh=df_rh.sort_values(by='datetime')
    df_rh=df_rh.reset_index(drop=True)
    
    
    df=pd.DataFrame()
    df['env_temp']=df_temp['value']
    df['env_rh']=df_rh['value']
    df['co2_co2']=df_co2['value']
    df['room']='roomX'
    
    unique_rooms = df['room'].unique()
    results = {}
    # model_name= os.getenv("s3://mlflow/15/7c347fc9147a4bc3a8ef6018e2d1a128/artifacts/BAG_univpm2") #uri model edf ML (BAG)
    model_name= "s3://mlflow/15/7c347fc9147a4bc3a8ef6018e2d1a128/artifacts/BAG_univpm2"
    try:
    # Load the MLflow model
        model = mlflow.sklearn.load_model(model_name)
        print("Loaded model:", model)
        print('model loaded')
        for room in unique_rooms:
                print('before room data')
                room_data = df[df['room'] == room][['env_temp', 'env_rh', 'co2_co2']] #, 'outdoorT', 'lux'
                
                print('before room predictions')
                predRoom=model.predict(room_data)
                print('before room mean')
                room_predictions = np.mean(predRoom)
                print('before results')
                results[room] = room_predictions.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
    return{'calculated comfort level': results}


@app.post("/predict_edf")
async def predict_comfort(): #file: bytes=File(...)
    pilotdb = "edf_test"
    #Temperature
    sensor_temp=['32852_0_84']
    #Humidity
    sensor_hum=['32840_0_72'] 
    #co2
    sensor_co2=['32835_0_67']
    
    #co2
    df_co2=execute_query(token, pilotdb, sensor_co2, time_from=None, time_to=None)
    df_co2['datetime']=pd.to_datetime(df_co2['datetime'], format="%Y%m%d%H%M%S")
    df_co2=df_co2.sort_values(by='datetime')
    df_co2=df_co2.reset_index(drop=True)
    # df_co2_res = np.interp(np.linspace(0, 1, len(df_temp)), np.linspace(0, 1, len(df_co2)), df_co2['value'])
    
    #temp
    df_temp=execute_query(token, pilotdb, sensor_temp, time_from=None, time_to=None)
    df_temp['datetime']=pd.to_datetime(df_temp['datetime'], format="%Y%m%d%H%M%S")
    df_temp=df_temp.sort_values(by='datetime')
    df_temp=df_temp.reset_index(drop=True)
    temps_res = np.interp(np.linspace(0, 1, len(df_co2)), np.linspace(0, 1, len(df_temp)), df_temp['value'])

    #rh
    df_rh=execute_query(token, pilotdb, sensor_hum, time_from=None, time_to=None)
    df_rh['datetime']=pd.to_datetime(df_rh['datetime'], format="%Y%m%d%H%M%S")
    df_rh=df_rh.sort_values(by='datetime')
    df_rh=df_rh.reset_index(drop=True)
    rh_res = np.interp(np.linspace(0, 1, len(df_co2)), np.linspace(0, 1, len(df_rh)), df_rh['value'])

    
    df1=pd.DataFrame()
    df1['env_temp']=temps_res#df_temp['value']
    df1['env_rh']=rh_res#df_rh['value']
    df1['co2_co2']=df_co2['value']
    df1['room']='roomX'
    
    unique_offices = df1['room'].unique()
    output = {}
    
    # model_name2=os.getenv("s3://mlflow/14/eedea11deed24d47a9d1c5536e73622d/artifacts/LSTM24_univpm") #uri model edf lstm
    model_name2="s3://mlflow/14/eedea11deed24d47a9d1c5536e73622d/artifacts/LSTM24_univpm"
    try:
       # Load the MLflow model
           model2 = mlflow.sklearn.load_model(model_name2) #
           for office in unique_offices:
               room_data = df1[df1['room'] == office][['env_temp', 'env_rh', 'co2_co2']] #, 'outdoorT', 'lux'

               #divido gli input in train e test
               train_size = int(len(room_data) * 0.8) 
               X_train, X_test = room_data[:train_size], room_data[train_size:]
               scaler = MinMaxScaler()
               X_train_scaled = scaler.fit_transform(X_train)
               X_test_scaled = scaler.fit_transform(X_test)
               # Create sequences for LSTM model
               sequence_length = 168  # 1 week (24h * 7)
               forecast_horizon = 24  # Predicting the next 24 hours
              
               #split the data (X,y) in train and test
               X_train_seq = create_sequences(X_train_scaled, sequence_length, forecast_horizon)
               X_test_seq = create_sequences(X_test_scaled, sequence_length, forecast_horizon)
                   
               room_comfort_pred = np.mean(model2.predict(X_test_seq))
               output[office] = room_comfort_pred.tolist()
                   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
    return{ 'predicted comfort level': output}


####FOCCHI
@app.post("/calculate_focchi")
async def calculate_comfort():  #environment: env_params, file: bytes=File(...)
    #richiesta datasharing per dati
    pilotdb = "focchi_test"
    #considero la sala panorai di focchi e nello specifico la zona 2 (perche abbiamo t, rh, co2)
    sensors_temp=[ 'z2_temp' ] #'z1_temp',
    sensors_rh=[ 'z2_rh' ] #'z1_rh',
    sensors_co2=['z2_co2' ]

    #temp
    df_temp=execute_query(token, pilotdb, sensors_temp, time_from=None, time_to=None)
    df_temp['datetime']=pd.to_datetime(df_temp['datetime'], format="%Y%m%d%H%M%S")
    df_temp=df_temp.sort_values(by='datetime')
    df_temp=df_temp.reset_index(drop=True)
    #rh
    df_rh=execute_query(token, pilotdb, sensors_rh, time_from=None, time_to=None)
    df_rh['datetime']=pd.to_datetime(df_rh['datetime'], format="%Y%m%d%H%M%S")
    df_rh=df_rh.sort_values(by='datetime')
    df_rh=df_rh.reset_index(drop=True)
    #co2
    df_co2=execute_query(token, pilotdb, sensors_co2, time_from=None, time_to=None)
    df_co2['datetime']=pd.to_datetime(df_co2['datetime'], format="%Y%m%d%H%M%S")
    df_co2=df_co2.sort_values(by='datetime')
    df_co2=df_co2.reset_index(drop=True)
    # df_co2_res = np.interp(np.linspace(0, 1, len(df_temp)), np.linspace(0, 1, len(df_co2)), df_co2['value'])
    
    df=pd.DataFrame()
    df['indoor T']=df_temp['value']
    df['indoor RH']=df_rh['value']
    df['CO2']=df_co2['value']
    df['room']='sala_panorami_z2'
    
    unique_rooms = df['room'].unique()
    results = {}
    # model_name= os.getenv("FOCCHI_CALCULATE_MODEL") 
    model_name='s3://mlflow/26/35954e9bbbd047458e20fb22b7d9af56/artifacts/BAG_univpm2'
    try:
    # Load the MLflow model
        model = mlflow.sklearn.load_model(model_name)
        print("Loaded model:", model)
        print('model loaded')
        for room in unique_rooms:
                print('before room data')
                room_data = df[df['room'] == room][['indoor T', 'indoor RH', 'CO2']] #, 'outdoorT', 'lux'
                
                print('before room predictions')
                predRoom=model.predict(room_data)
                print('before room mean')
                room_predictions = np.mean(predRoom)
                print('before results')
                results[room] = room_predictions.tolist()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
    return{'calculated comfort level': results}


@app.post("/predict_focchi")
async def predict_comfort(): 
    pilotdb = "focchi_test"
    sensors_temp=[ 'z2_temp' ] 
    sensors_rh=[ 'z2_rh' ] 
    sensors_co2=['z2_co2' ]

    #temp
    df_temp=execute_query(token, pilotdb, sensors_temp, time_from=None, time_to=None)
    df_temp['datetime']=pd.to_datetime(df_temp['datetime'], format="%Y%m%d%H%M%S")
    df_temp=df_temp.sort_values(by='datetime')
    df_temp=df_temp.reset_index(drop=True)
    #rh
    df_rh=execute_query(token, pilotdb, sensors_rh, time_from=None, time_to=None)
    df_rh['datetime']=pd.to_datetime(df_rh['datetime'], format="%Y%m%d%H%M%S")
    df_rh=df_rh.sort_values(by='datetime')
    df_rh=df_rh.reset_index(drop=True)
    #co2
    df_co2=execute_query(token, pilotdb, sensors_co2, time_from=None, time_to=None)
    df_co2['datetime']=pd.to_datetime(df_co2['datetime'], format="%Y%m%d%H%M%S")
    df_co2=df_co2.sort_values(by='datetime')
    df_co2=df_co2.reset_index(drop=True)
    # df_co2_res = np.interp(np.linspace(0, 1, len(df_temp)), np.linspace(0, 1, len(df_co2)), df_co2['value'])
    
    df1=pd.DataFrame()
    df1['indoor T']=df_temp['value']
    df1['indoor RH']=df_rh['value']
    df1['CO2']=df_co2['value']
    df1['room']='sala_panorami_z2'
    
    unique_offices = df1['room'].unique()
    output = {}
    
    # model_name2=os.getenv("FOCCHI_PREDICT_MODEL")
    model_name2='s3://mlflow/27/f9c766a2e5024093b4d22b548bb61b1d/artifacts/LSTM24_univpm_reduced'
    try:
       # Load the MLflow model
           model2 = mlflow.sklearn.load_model(model_name2) #
           for office in unique_offices:
               room_data = df1[df1['room'] == office][['indoor T', 'indoor RH', 'CO2']] #, 'outdoorT', 'lux'

               #divido gli input in train e test
               train_size = int(len(room_data) * 0.8) 
               X_train, X_test = room_data[:train_size], room_data[train_size:]
               scaler = MinMaxScaler()
               X_train_scaled = scaler.fit_transform(X_train)
               X_test_scaled = scaler.fit_transform(X_test)
               # Create sequences for LSTM model
               sequence_length = 168  # 1 week (24h * 7)
               forecast_horizon = 24  # Predicting the next 24 hours
              
               #split the data (X,y) in train and test
               X_train_seq = create_sequences(X_train_scaled, sequence_length, forecast_horizon)
               X_test_seq = create_sequences(X_test_scaled, sequence_length, forecast_horizon)
                   
               room_comfort_pred = np.mean(model2.predict(X_test_seq))
               output[office] = room_comfort_pred.tolist()
                   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
    return{ 'predicted comfort level': output}

####FOCCHI updated RH Tin (...updating...)
@app.post("/predict_focchi_updated")
async def predict_comfort(): 
    pilotdb = "focchi_test"
    sensors_temp=[ 'z2_temp' ] 
    sensors_rh=[ 'z2_rh' ] 
    sensors_co2=['z2_co2' ]
    sensors_out=['']
    #temp
    df_temp=execute_query(token, pilotdb, sensors_temp, time_from=None, time_to=None)
    df_temp['datetime']=pd.to_datetime(df_temp['datetime'], format="%Y%m%d%H%M%S")
    df_temp=df_temp.sort_values(by='datetime')
    df_temp=df_temp.reset_index(drop=True)
    #rh
    df_rh=execute_query(token, pilotdb, sensors_rh, time_from=None, time_to=None)
    df_rh['datetime']=pd.to_datetime(df_rh['datetime'], format="%Y%m%d%H%M%S")
    df_rh=df_rh.sort_values(by='datetime')
    df_rh=df_rh.reset_index(drop=True)
    #co2
    df_co2=execute_query(token, pilotdb, sensors_co2, time_from=None, time_to=None)
    df_co2['datetime']=pd.to_datetime(df_co2['datetime'], format="%Y%m%d%H%M%S")
    df_co2=df_co2.sort_values(by='datetime')
    df_co2=df_co2.reset_index(drop=True)
    # df_co2_res = np.interp(np.linspace(0, 1, len(df_temp)), np.linspace(0, 1, len(df_co2)), df_co2['value'])
    
    # #outdoor
    # for filename in ['weather_univpm.json']:
    #     file_path = os.path.join(directory, filename)
    #     # Read the JSON file
    #     with open(file_path, 'r') as file:
    #         data = json.load(file)
    #         use_cases.extend(data)

    # # Generate token
    # token = gen_token(username, password)

    # # Process each input
    # for use_case in use_cases:
    #     df = process_in_batches(use_case, token)
    #     print(df)

    # df_res=df.loc[df['location_id']==6]
    
    # df_weather=execute_query(token, pilotdb, sensors_out, time_from=None, time_to=None)
    # df_weather['datetime']=pd.to_datetime(df_weather['datetime'], format="%Y%m%d%H%M%S")
    # df_weather=df_weather.sort_values(by='datetime')
    # df_weather=df_weather.reset_index(drop=True)
    # #outnext
    # outNext=df_weather['outdoorT'].shift(-24)
    # dataset['outNext']=outNext
    # #forecast weather 
    # # https://api.weatherbit.io/v2.0/forecast/hourly?city=Raleigh,NC&key=API_KEY&hours=48

    #dividere input dataset giorno per giorno 
    dataset_crop=dataset[0:42469]
    dataset_crop['day_of_week_sin'] = np.sin(2 * np.pi * dataset_crop['day_of_week']/7)
    dataset_crop['day_of_week_cos'] = np.cos(2 * np.pi * dataset_crop['day_of_week']/7)
    # Supponiamo che tu abbia una colonna 'hour' che va da 0 a 23
    dataset_crop['hour_sin'] = np.sin(2 * np.pi * dataset_crop['hour']/24)
    dataset_crop['hour_cos'] = np.cos(2 * np.pi * dataset_crop['hour']/24)
    
    
    df1=pd.DataFrame()
    df1['indoor T']=df_temp['value']
    df1['indoor RH']=df_rh['value']
    df1['CO2']=df_co2['value']
    df1['outdoor T']=df_weather['value'] #missing
    df1['outNext']=dataset['outNext']
    df1['day_of_week_sin']=dataset_crop['day_of_week_sin']
    df1['day_of_week_cos']=dataset_crop['day_of_week_cos']
    df1['hour_sin']=dataset_crop['hour_sin']
    df1['hour_cos']=dataset_crop['hour_cos']
    
    df1['room']='sala_panorami_z2'
    
    unique_offices = df1['room'].unique()
    output = {}
    # model_name2=os.getenv("FOCCHI_PREDICT_MODEL")
    model_name2='s3://mlflow/40/57affbbe23a04c5f87a842c60608ff38/artifacts/LSTM1_univpm_Tin12' #model Tin
    model_name3='' #model rh
    try:
       # Load the MLflow model
           model2 = mlflow.sklearn.load_model(model_name2) #
           model3= mlflow.sklearn.load_model(model_name3)
           for office in unique_offices:
               room_data = df1[df1['room'] == office][['indoor T', 'outdoor T', 'outNext', 'day_of_week_sin','day_of_week_cos','hour_sin', 'hour_cos']] #, 'outdoorT', 'lux'
               room_data2 = df1[df1['room'] == office][['indoor RH', 'outdoor T', 'outNext', 'day_of_week_sin','day_of_week_cos','hour_sin', 'hour_cos']] #, 'outdoorT', 'lux'

               #divido gli input in train e test -Tin
               train_size = int(len(room_data) * 0.8) 
               X_train, X_test = room_data[:train_size], room_data[train_size:]
               scaler = MinMaxScaler()
               X_train_scaled = scaler.fit_transform(X_train)
               X_test_scaled = scaler.fit_transform(X_test)
               #divido gli input in train e test -RH
               train_sizerh = int(len(room_data2) * 0.8) 
               X_trainrh, X_testrh = room_data2[:train_sizerh], room_data2[train_sizerh:]
               X_train_scaledrh = scaler.fit_transform(X_trainrh)
               X_test_scaledrh = scaler.fit_transform(X_testrh)
               # Create sequences for LSTM model
               sequence_length = 168  # 1 week (24h * 7)
               forecast_horizon = 12  # Predicting the next 24 hours
               #split the data (X,y) in train and test -tin
               X_train_seq = create_sequences(X_train_scaled, sequence_length, forecast_horizon)
               X_test_seq = create_sequences(X_test_scaled, sequence_length, forecast_horizon)
               #split the data (X,y) in train and test -rh
               X_train_seqrh = create_sequences(X_train_scaledrh, sequence_length, forecast_horizon)
               X_test_seqrh = create_sequences(X_test_scaledrh, sequence_length, forecast_horizon)
               room_tin_pred=model2.predict(X_test_seq)#np.mean(
               room_rh_pred=model3.predict(X_test_seqrh)
               #comfort computation with future level of Tin and RH
               room_comfort_pred = np.mean(spmv_calculation(room_tin_pred, room_rh_pred, office, df1['date']))#tocheck #indoorT, indoorRH, sensor, date
               output[office] = room_comfort_pred.tolist()
                   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
    return{ 'predicted comfort level 12h': output}

####IEECP
@app.post("/calculate_ieecp")
async def calculate_comfort(): 
    #richiesta datasharing per dati
    pilotdb = "ieecp_test"
    sensors_temp=[ '70:ee:50:96:1f:fa_temperature' ] 
    sensors_rh=[ '70:ee:50:96:1f:fa_humidity' ]
    sensors_co2=['70:ee:50:96:1f:fa_co2' ]

    #temp
    df_temp=execute_query(token, pilotdb, sensors_temp, time_from=None, time_to=None)
    df_temp['datetime']=pd.to_datetime(df_temp['datetime'], format="%Y%m%d%H%M%S")
    df_temp=df_temp.sort_values(by='datetime')
    df_temp=df_temp.reset_index(drop=True)
    #rh
    df_rh=execute_query(token, pilotdb, sensors_rh, time_from=None, time_to=None)
    df_rh['datetime']=pd.to_datetime(df_rh['datetime'], format="%Y%m%d%H%M%S")
    df_rh=df_rh.sort_values(by='datetime')
    df_rh=df_rh.reset_index(drop=True)
    #co2
    df_co2=execute_query(token, pilotdb, sensors_co2, time_from=None, time_to=None)
    df_co2['datetime']=pd.to_datetime(df_co2['datetime'], format="%Y%m%d%H%M%S")
    df_co2=df_co2.sort_values(by='datetime')
    df_co2=df_co2.reset_index(drop=True)
    # df_co2_res = np.interp(np.linspace(0, 1, len(df_temp)), np.linspace(0, 1, len(df_co2)), df_co2['value'])
    
    df=pd.DataFrame()
    df['indoorT']=df_temp['value']
    df['indoorRH']=df_rh['value']
    df['co2']=df_co2['value']
    df['room']='xxx'
    
    unique_rooms = df['room'].unique()
    results = {}
    # model_name= os.getenv("") #MLflow uri ieecp ML (BAG)
    
    model_name='s3://mlflow/7/1735aa7d11464516bfe6d7813148cbdb/artifacts/BAG_univpm2'
    
    try:
    # Load the MLflow model
        model = mlflow.sklearn.load_model(model_name)
        print("Loaded model:", model)
        print('model loaded')
        for room in unique_rooms:
                print('before room data')
                room_data = df[df['room'] == room][['indoorT', 'indoorRH', 'co2']] #, 'outdoorT', 'lux'
                
                print('before room predictions')
                predRoom=model.predict(room_data)
                print('before room mean')
                room_predictions = np.mean(predRoom)
                print('before results')
                results[room] = room_predictions.tolist()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
    return{'calculated comfort level': results}


@app.post("/predict_ieecp")
async def predict_comfort(): 
    pilotdb = "ieecp_test"
    sensors_temp=[ '70:ee:50:96:1d:fc_temperature' ] 
    sensors_rh=[ '70:ee:50:96:1d:fc_humidity' ]
    sensors_co2=['70:ee:50:96:1d:fc_co2' ]

    #temp
    df_temp=execute_query(token, pilotdb, sensors_temp, time_from=None, time_to=None)
    df_temp['datetime']=pd.to_datetime(df_temp['datetime'], format="%Y%m%d%H%M%S")
    df_temp=df_temp.sort_values(by='datetime')
    df_temp=df_temp.reset_index(drop=True)
    #rh
    df_rh=execute_query(token, pilotdb, sensors_rh, time_from=None, time_to=None)
    df_rh['datetime']=pd.to_datetime(df_rh['datetime'], format="%Y%m%d%H%M%S")
    df_rh=df_rh.sort_values(by='datetime')
    df_rh=df_rh.reset_index(drop=True)
    #co2
    df_co2=execute_query(token, pilotdb, sensors_co2, time_from=None, time_to=None)
    df_co2['datetime']=pd.to_datetime(df_co2['datetime'], format="%Y%m%d%H%M%S")
    df_co2=df_co2.sort_values(by='datetime')
    df_co2=df_co2.reset_index(drop=True)
    # df_co2_res = np.interp(np.linspace(0, 1, len(df_temp)), np.linspace(0, 1, len(df_co2)), df_co2['value'])
    
    df1=pd.DataFrame()
    df1['indoorT']=df_temp['value']
    df1['indoorRH']=df_rh['value']
    df1['co2']=df_co2['value']
    df1['room']='xxx'
    
    unique_offices = df1['room'].unique()
    output = {}
    
    # model_name2=os.getenv("") #MLflow uri for LSTM ieecp 
    model_name2='s3://mlflow/6/72368d8d783544e3bacfde3795fb7af7/artifacts/LSTM24_univpm'
    
    try:
       # Load the MLflow model
           model2 = mlflow.sklearn.load_model(model_name2) #
           for office in unique_offices:
               room_data = df1[df1['room'] == office][['indoorT', 'indoorRH', 'co2']] #, 'outdoorT', 'lux'

               #divido gli input in train e test
               train_size = int(len(room_data) * 0.8) 
               X_train, X_test = room_data[:train_size], room_data[train_size:]
               scaler = MinMaxScaler()
               X_train_scaled = scaler.fit_transform(X_train)
               X_test_scaled = scaler.fit_transform(X_test)
               # Create sequences for LSTM model
               sequence_length = 168  # 1 week (24h * 7)
               forecast_horizon = 24  # Predicting the next 24 hours
              
               #split the data (X,y) in train and test
               X_train_seq = create_sequences(X_train_scaled, sequence_length, forecast_horizon)
               X_test_seq = create_sequences(X_test_scaled, sequence_length, forecast_horizon)
                   
               room_comfort_pred = np.mean(model2.predict(X_test_seq))
               output[office] = room_comfort_pred.tolist()
                   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
    return{ 'predicted comfort level': output}

####FVH
@app.post("/calculate_fvh")
async def calculate_comfort(): 
    #richiesta datasharing per dati
    pilotdb = "fvh_test"
    sensors_temp=[ '144858', '144864' ] 
    sensors_rh=[ '144855', '144860' ] 
    sensors_co2=['144856', '144861' ]
    
    #temp
    df_temp=execute_query(token, pilotdb, sensors_temp, time_from=None, time_to=None)
    df_temp['datetime']=pd.to_datetime(df_temp['datetime'], format="%Y%m%d%H%M%S")
    df_temp=df_temp.sort_values(by='datetime')
    df_temp=df_temp.reset_index(drop=True)
    #rh
    df_rh=execute_query(token, pilotdb, sensors_rh, time_from=None, time_to=None)
    df_rh['datetime']=pd.to_datetime(df_rh['datetime'], format="%Y%m%d%H%M%S")
    df_rh=df_rh.sort_values(by='datetime')
    df_rh=df_rh.reset_index(drop=True)
    #co2
    df_co2=execute_query(token, pilotdb, sensors_co2, time_from=None, time_to=None)
    df_co2['datetime']=pd.to_datetime(df_co2['datetime'], format="%Y%m%d%H%M%S")
    df_co2=df_co2.sort_values(by='datetime')
    df_co2=df_co2.reset_index(drop=True)
    # df_co2_res = np.interp(np.linspace(0, 1, len(df_temp)), np.linspace(0, 1, len(df_co2)), df_co2['value'])
    
    df=pd.DataFrame()
    df['indoor T']=df_temp['value']
    df['indoor RH']=df_rh['value']
    df['co2']=df_co2['value']
    df['room']='xxx'
    
    unique_rooms = df['room'].unique()
    results = {}
    # model_name= os.getenv("s3://mlflow/20/09aa687c59164cd299002d035bc5fdbf/artifacts/BAG_univpm") #MLflow uri fvh ML (BAG)
    model_name= "s3://mlflow/23/dc4e7003811f4e20bb2116547efe094a/artifacts/BAG_univpm"
    try:
    # Load the MLflow model
        model = mlflow.sklearn.load_model(model_name)
        print("Loaded model:", model)
        print('model loaded')
        for room in unique_rooms:
                print('before room data')
                room_data = df[df['room'] == room][['indoor T', 'indoor RH', 'co2']] #, 'outdoorT', 'lux'
                
                print('before room predictions')
                predRoom=model.predict(room_data)
                print('before room mean')
                room_predictions = np.mean(predRoom)
                print('before results')
                results[room] = room_predictions.tolist()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
    return{'calculated comfort level': results}

@app.post("/predict_fvh")
async def predict_comfort(): 
    pilotdb = "fvh_test"
    sensors_temp=[ '144858', '144864' ] 
    sensors_rh=[ '144855', '144860' ] 
    sensors_co2=['144856', '144861' ]

    #temp
    df_temp=execute_query(token, pilotdb, sensors_temp, time_from=None, time_to=None)
    df_temp['datetime']=pd.to_datetime(df_temp['datetime'], format="%Y%m%d%H%M%S")
    df_temp=df_temp.sort_values(by='datetime')
    df_temp=df_temp.reset_index(drop=True)
    #rh
    df_rh=execute_query(token, pilotdb, sensors_rh, time_from=None, time_to=None)
    df_rh['datetime']=pd.to_datetime(df_rh['datetime'], format="%Y%m%d%H%M%S")
    df_rh=df_rh.sort_values(by='datetime')
    df_rh=df_rh.reset_index(drop=True)
    #co2
    df_co2=execute_query(token, pilotdb, sensors_co2, time_from=None, time_to=None)
    df_co2['datetime']=pd.to_datetime(df_co2['datetime'], format="%Y%m%d%H%M%S")
    df_co2=df_co2.sort_values(by='datetime')
    df_co2=df_co2.reset_index(drop=True)
    # df_co2_res = np.interp(np.linspace(0, 1, len(df_temp)), np.linspace(0, 1, len(df_co2)), df_co2['value'])
    
    df1=pd.DataFrame()
    
    df1['indoor T']=df_temp['value']
    df1['indoor RH']=df_rh['value']
    df1['co2']=df_co2['value']
    df1['room']='xxx'
    
    unique_offices = df1['room'].unique()
    output = {}
    
    # model_name2=os.getenv("") #MLflow uri for LSTM fvh 
    model_name2= "s3://mlflow/22/183d7500f2db4479b864aff7c6e63070/artifacts/LSTM24_univpm"
    try:
       # Load the MLflow model
           model2 = mlflow.sklearn.load_model(model_name2) #
           for office in unique_offices:
               room_data = df1[df1['room'] == office][['indoor T', 'indoor RH', 'co2']] #, 'outdoorT', 'lux'

               #divido gli input in train e test
               train_size = int(len(room_data) * 0.8) 
               X_train, X_test = room_data[:train_size], room_data[train_size:]
               scaler = MinMaxScaler()
               X_train_scaled = scaler.fit_transform(X_train)
               X_test_scaled = scaler.fit_transform(X_test)
               # Create sequences for LSTM model
               sequence_length = 168  # 1 week (24h * 7)
               forecast_horizon = 24  # Predicting the next 24 hours
              
               #split the data (X,y) in train and test
               X_train_seq = create_sequences(X_train_scaled, sequence_length, forecast_horizon)
               X_test_seq = create_sequences(X_test_scaled, sequence_length, forecast_horizon)
                   
               room_comfort_pred = np.mean(model2.predict(X_test_seq))
               output[office] = room_comfort_pred.tolist()
                   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
    return{ 'predicted comfort level': output}


if __name__ == "__main__":
   
    import uvicorn
    uvicorn.run("main_missingP:app", host="0.0.0.0", port=5000, reload=True)



#backup code for FVH

# verbose = True
# selected_columns2 = ['indoor T','indoor RH', 'co2', 'outT', 'room']
# # InfluxDB 2.0 Credentials
#influx_url = "https://influx.fvh.io"
#influx_org = "Energy"
#influx_bucket = "Nuuka_7683"
# influx_token = "S4T_4IPUUnugYf3ea15r1fne3NDwHeyCVav4EuUnSmFX4g49dmR54GUU0_xpnGV5j8Wv6XrPA-aLCEPTwewGng=="
# # Create InfluxDB client. Use enable_gzip=True to enable gzip compression and reduce download times.
# client = InfluxDBClient(url=influx_url, token=influx_token, enable_gzip=True)
# @app.get("/")
# async def root():
#     return{"message":"hi"} 
# @app.post("/fvh_calculate")
# async def calculate_comfort():  #environment: env_params , file: bytes=File(...)
#     #carico csv su fastapi
#     if verbose: 
#         print ("taking measurements...")

#     query_measurements()

#     if verbose: 
#         print ("taking meta data...")
        
#     df_meta_original = query_meta()
    
#     df_meta=df_meta_original[['category', 'datapointid', 'Room']]

#     #df_meta.to_excel("meta_data.xlsx")
#     #print(df_meta)
#     #print(df_meta['category'].unique())
#     #print(df_meta.columns)
#     #print(df_meta['datapointid'].values.tolist())
    
#     #TEMPERATURE
#     # Filter out only the rows that have category == "indoor conditions: temperatur"
#     df_meta_temperature = df_meta[df_meta["category"] == "indoor conditions: temperature"].copy()
#     df_meta_temperature.reset_index(drop=True)
#     df_meta_temperature.dropna(subset=['Room'], inplace=True)
#     df_meta_temperature.drop(df_meta_temperature[df_meta_temperature.duplicated(subset='Room', keep=False)].index, inplace=True)
#     df_meta_temperature.reset_index(drop=True, inplace=True)
#     df_meta_temperature['datapointid'] = df_meta_temperature['datapointid'].astype(int)
#     # df_meta_temperature=df_meta_temperature[0:5]

#     # Get 10 first values from datapointid column to a list by getting 10 first
#     # rows and then getting the values from the column
#     total_rows = len(df_meta_temperature)

#     # Step 2: Initialize an empty DataFrame to store the combined results
#     temperature_df = pd.DataFrame()

#     # Step 3: Iterate over every 10 rows and append the results of 'query_data' to the 'result_df'
#     if verbose:
#         print('taking temperatures value')
#     for i in range(0, total_rows, 10):
#         datapointids = df_meta_temperature["datapointid"].iloc[i:i+10].values.tolist()
#         # if verbose:
#         #     print('iter ' + str(i) + ' size: ' + str(len(datapointids)))
#         data = query_data(datapointids)
#         temperature_df = pd.concat([temperature_df, data])
#     if verbose:
#         print('step 2')
#     temperature_df['datetime'] = temperature_df.index
#     temperature_df.reset_index(drop=True, inplace=True)
#     temperature_df['datapointid'] = temperature_df['datapointid'].astype(int)
#     merged_df = temperature_df.merge(df_meta[['datapointid', 'Room']], on='datapointid', how='left') #, 'Floor', 'FloorSection'
#     if verbose:
#         print('making df')
#     # Create the final dataframe with the desired structure
#     temperature_df = pd.DataFrame({
#         'DATE': merged_df['datetime'],
#         'room': merged_df['Room'],
#         # 'floor': merged_df['Floor'],
#         # 'section': merged_df['FloorSection'],
#         'indoor T': merged_df['value'],
#     })
#     #CO2
#     df_meta_co2 = df_meta[df_meta["category"] == "indoor conditions: co2"].copy()
#     df_meta_co2.reset_index(drop=True)
#     df_meta_co2.dropna(subset=['Room'], inplace=True)
#     df_meta_co2.drop(df_meta_co2[df_meta_co2.duplicated(subset='Room', keep=False)].index, inplace=True)
#     df_meta_co2.reset_index(drop=True, inplace=True)
#     df_meta_co2['datapointid'] = df_meta_co2['datapointid'].astype(int)
#     # df_meta_co2=df_meta_co2[0:5]

#     total_rows = len(df_meta_co2)

#     # Step 2: Initialize an empty DataFrame to store the combined results
#     co2_df = pd.DataFrame()

#     # Step 3: Iterate over every 10 rows and append the results of 'query_data' to the 'result_df'
#     for i in range(0, total_rows, 10):
#         datapointids = df_meta_co2["datapointid"].iloc[i:i+10].values.tolist()
#         data = query_data(datapointids)
#         co2_df = pd.concat([co2_df, data])

#     co2_df['datetime'] = co2_df.index
#     co2_df.reset_index(drop=True, inplace=True)
#     co2_df['datapointid'] = co2_df['datapointid'].astype(int)
#     merged_df = co2_df.merge(df_meta[['datapointid', 'Room']], on='datapointid', how='left') #, 'Floor', 'FloorSection'

#     # Create the final dataframe with the desired structure
#     co2_df = pd.DataFrame({
#         'DATE': merged_df['datetime'],
#         'room': merged_df['Room'],
#         'co2': merged_df['value'],
#     })

#     #RELATIVE HUMIDITY

#     df_meta_rh = df_meta[df_meta["category"] == "indoor conditions: relative humidity"].copy()
#     df_meta_rh.reset_index(drop=True)
#     df_meta_rh.dropna(subset=['Room'], inplace=True)
#     df_meta_rh.drop(df_meta_rh[df_meta_rh.duplicated(subset='Room', keep=False)].index, inplace=True)
#     df_meta_rh.reset_index(drop=True, inplace=True)
#     df_meta_rh['datapointid'] = df_meta_rh['datapointid'].astype(int)
#     # df_meta_rh=df_meta_rh[0:5]

#     total_rows = len(df_meta_rh)

#     # Step 2: Initialize an empty DataFrame to store the combined results
#     rh_df = pd.DataFrame()

#     # Step 3: Iterate over every 10 rows and append the results of 'query_data' to the 'result_df'
#     for i in range(0, total_rows, 10):
#         datapointids = df_meta_rh["datapointid"].iloc[i:i+10].values.tolist()
#         data = query_data(datapointids)
#         rh_df = pd.concat([rh_df, data])
#     rh_df['datetime'] = rh_df.index
#     rh_df.reset_index(drop=True, inplace=True)
#     rh_df['datapointid'] = rh_df['datapointid'].astype(int)
    
#     merged_df = rh_df.merge(df_meta[['datapointid', 'Room']], on='datapointid', how='left') #, 'Floor', 'FloorSection'

#     # Create the final dataframe with the desired structure
#     rh_df = pd.DataFrame({
#         'DATE': merged_df['datetime'],
#         'room': merged_df['Room'],
#         'indoor RH': merged_df['value'],
#     })

#     #OUTT
#     df_meta_outtemperature = df_meta[df_meta["category"] == "outdoor conditions: outdoor temperature, measured at location"].copy()
#     df_meta_outtemperature.reset_index(drop=True)
#     df_meta_outtemperature.dropna(subset=['Room'], inplace=True)
#     df_meta_outtemperature.drop(df_meta_outtemperature[df_meta_outtemperature.duplicated(subset='Room', keep=False)].index, inplace=True)
#     df_meta_outtemperature.reset_index(drop=True, inplace=True)
#     df_meta_outtemperature['datapointid'] = df_meta_outtemperature['datapointid'].astype(int)
#     # df_meta_outtemperature=df_meta_outtemperature[0:5]

#     # Get 10 first values from datapointid column to a list by getting 10 first
#     # rows and then getting the values from the column
#     total_rows = len(df_meta_outtemperature)

#     # Step 2: Initialize an empty DataFrame to store the combined results
#     outtemperature_df = pd.DataFrame()

#     # Step 3: Iterate over every 10 rows and append the results of 'query_data' to the 'result_df'
#     for i in range(0, total_rows, 10):
#         datapointids = df_meta_outtemperature["datapointid"].iloc[i:i+10].values.tolist()
#         data = query_data(datapointids)
#         outtemperature_df = pd.concat([outtemperature_df, data])

#     outtemperature_df['datetime'] = outtemperature_df.index
#     outtemperature_df.reset_index(drop=True, inplace=True)
#     outtemperature_df['datapointid'] = outtemperature_df['datapointid'].astype(int)
#     merged_df = outtemperature_df.merge(df_meta[['datapointid', 'Room']], on='datapointid', how='left') #, 'Floor', 'FloorSection'

#     # Create the final dataframe with the desired structure
#     outtemperature_df = pd.DataFrame({
#         'DATE': merged_df['datetime'],
#         'room': merged_df['Room'],
#         # 'floor': merged_df['Floor'],
#         # 'section': merged_df['FloorSection'],
#         'outT': merged_df['value'],
#     })

#     outtemperature_df=outtemperature_df.sort_values(by='DATE')
#     # outtemperature_df.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/fvh pilot/outdoor_FVH_260224.csv')

#     # df_new=[]

#     ####concat and create the merged dataset
#     result_df = pd.concat([co2_df, temperature_df, rh_df], ignore_index=True)
#     result_df_sorted = result_df.sort_values(by='DATE')
#     merged_df = pd.merge(temperature_df, rh_df, on=['DATE', 'room'])
#     merged_df = pd.merge(merged_df, co2_df, on=['DATE', 'room'])
#     # merged_df = pd.merge(merged_df, outtemperature_df, on=['DATE', 'room'])
#     merged_df=merged_df.sort_values(by=['DATE'])
#     merged_df=merged_df.reset_index(drop=True)

#     #aggiungere l'outdoor ai parametri indoor collected dalle room di interesse (25/33)
#     new_out=[]
#     room_label=merged_df['room'].unique()
#     dataallrooms=[]
#     for j in range(len(room_label)):
#         dataset_roomx=merged_df.loc[merged_df['room']==room_label[j]]
#         dataset_roomx=dataset_roomx.reset_index(drop=True)
#         dataset_roomx=dataset_roomx.sort_values(by='DATE')
#         #la temp outdoor è stata collected da 3 stanze diverse da quelle dell'indoor. Ho selezionato
#         #una delle tre stanze tanto le temp esterne sono uguali
#         out_T310=outtemperature_df.loc[outtemperature_df['room']==str(310)]
#         out_T310=out_T310.sort_values(by='DATE')
#         out_T310=out_T310.reset_index(drop=True)
#         #sincronizzazione delle date
#         #1) elimino dalle temp out tutti i campioni presi in date out fo range delle date degli indoor
#         enddate_ind=dataset_roomx['DATE'].max()
#         startdate_ind=dataset_roomx['DATE'].min()
#         enddate_out=out_T310['DATE'].max()
#         startdate_out=out_T310['DATE'].min()
#         #outdoor filtrato
#         filtered_outT = out_T310[(out_T310['DATE'] <= enddate_ind) | (out_T310['DATE'] >= startdate_ind)]
#         #elimino tutte quelle date che non sono in comune tra i due dataset (outdoor e indoor)
#         df_out=filtered_outT[filtered_outT.DATE.isin(dataset_roomx.DATE)]    
#         df2=dataset_roomx[dataset_roomx.DATE.isin(df_out.DATE)]
#         df_out=df_out.reset_index(drop=True)
#         df2=df2.reset_index(drop=True)
#         df2['outT']=df_out['outT']
#         dataallrooms.append(df2)
#         # new_out.append(df_out)

#     df3 = pd.concat(dataallrooms, ignore_index=True)
#     df3=df3.reset_index(drop=True)

#     #PER API
#     # s3=str(file, 'utf-8')
#     # data3=StringIO(s3)
#     # df3=pd.read_csv(data3, usecols=selected_columns2)    
#     X2 = df3[['indoor T','indoor RH', 'co2', 'outT']]
#     lst2=X2.values.tolist()
   
#     # Assuming 'room' is a column in your DataFrame
#     unique_rooms = df3['room'].unique()

#     results = {}
#     model_name3='s3://mlflow/60/db3a6158b1da4f2884e5b939b2a0dccc/artifacts/BAG_univpm'

#     try:
#     # Load the MLflow model
    
#         model3 = mlflow.sklearn.load_model(model_name3)
#         i=1
#         for i in range(len(unique_rooms)):
#                 room_data = df3[df3['room'] == unique_rooms[i]][['indoor T', 'indoor RH', 'co2', 'outT']]
#                 if verbose: 
#                     print ("predicting " + str(i) + "...")
#                 room_predictions = np.mean(model3.predict(room_data))
#                 results[str(unique_rooms[i])] = room_predictions.tolist()

    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
#     return{'calculated comfort level for each room': results}


# @app.post("/fvh_predict")
# async def predict_comfort(): #file: bytes=File(...)
#     #carico il csv su fastapi
#     if verbose: 
#         print ("taking measurements...")

#     query_measurements()
#     if verbose: 
#         print ("taking meta data...")

#     df_meta = query_meta()
#     #df_meta.to_excel("meta_data.xlsx")
#     #print(df_meta)
#     #print(df_meta['category'].unique())
#     #print(df_meta.columns)
#     #print(df_meta['datapointid'].values.tolist())

#     #TEMPERATURE
#     # Filter out only the rows that have category == "indoor conditions: temperatur"
#     df_meta_temperature = df_meta[df_meta["category"] == "indoor conditions: temperature"].copy()
#     df_meta_temperature.reset_index(drop=True)
#     df_meta_temperature.dropna(subset=['Room'], inplace=True)
#     df_meta_temperature.drop(df_meta_temperature[df_meta_temperature.duplicated(subset='Room', keep=False)].index, inplace=True)
#     df_meta_temperature.reset_index(drop=True, inplace=True)
#     df_meta_temperature['datapointid'] = df_meta_temperature['datapointid'].astype(int)

#     # Get 10 first values from datapointid column to a list by getting 10 first
#     # rows and then getting the values from the column
#     total_rows = len(df_meta_temperature)

#     # Step 2: Initialize an empty DataFrame to store the combined results
#     temperature_df = pd.DataFrame()
#     if verbose:
#         print('taking temperatures value')

#     # Step 3: Iterate over every 10 rows and append the results of 'query_data' to the 'result_df'
#     for i in range(0, total_rows, 10):

#         datapointids = df_meta_temperature["datapointid"].iloc[i:i+10].values.tolist()
#         if verbose:
#             print('iter ' + str(i) + ' size: ' + str(len(datapointids)))

#         data = query_data(datapointids)
#         temperature_df = pd.concat([temperature_df, data])

#     temperature_df['datetime'] = temperature_df.index
#     temperature_df.reset_index(drop=True, inplace=True)
#     temperature_df['datapointid'] = temperature_df['datapointid'].astype(int)
#     merged_df = temperature_df.merge(df_meta[['datapointid', 'Room', 'Floor', 'FloorSection']], on='datapointid', how='left')

#     # Create the final dataframe with the desired structure
#     temperature_df = pd.DataFrame({
#         'DATE': merged_df['datetime'],
#         'room': merged_df['Room'],
#         'floor': merged_df['Floor'],
#         'section': merged_df['FloorSection'],
#         'indoor T': merged_df['value'],
#     })
#     #CO2

#     df_meta_co2 = df_meta[df_meta["category"] == "indoor conditions: co2"].copy()
#     df_meta_co2.reset_index(drop=True)
#     df_meta_co2.dropna(subset=['Room'], inplace=True)
#     df_meta_co2.drop(df_meta_co2[df_meta_co2.duplicated(subset='Room', keep=False)].index, inplace=True)
#     df_meta_co2.reset_index(drop=True, inplace=True)
#     df_meta_co2['datapointid'] = df_meta_co2['datapointid'].astype(int)

#     total_rows = len(df_meta_co2)

#     # Step 2: Initialize an empty DataFrame to store the combined results
#     co2_df = pd.DataFrame()

#     # Step 3: Iterate over every 10 rows and append the results of 'query_data' to the 'result_df'
#     for i in range(0, total_rows, 10):
#         datapointids = df_meta_co2["datapointid"].iloc[i:i+10].values.tolist()
#         data = query_data(datapointids)
#         co2_df = pd.concat([co2_df, data])

#     co2_df['datetime'] = co2_df.index
#     co2_df.reset_index(drop=True, inplace=True)
#     co2_df['datapointid'] = co2_df['datapointid'].astype(int)
#     merged_df = co2_df.merge(df_meta[['datapointid', 'Room', 'Floor', 'FloorSection']], on='datapointid', how='left')

#     # Create the final dataframe with the desired structure
#     co2_df = pd.DataFrame({
#         'DATE': merged_df['datetime'],
#         'room': merged_df['Room'],
#         'co2': merged_df['value'],
#     })

#     #RELATIVE HUMIDITY

#     df_meta_rh = df_meta[df_meta["category"] == "indoor conditions: relative humidity"].copy()
#     df_meta_rh.reset_index(drop=True)
#     df_meta_rh.dropna(subset=['Room'], inplace=True)
#     df_meta_rh.drop(df_meta_rh[df_meta_rh.duplicated(subset='Room', keep=False)].index, inplace=True)
#     df_meta_rh.reset_index(drop=True, inplace=True)
#     df_meta_rh['datapointid'] = df_meta_rh['datapointid'].astype(int)

#     total_rows = len(df_meta_rh)

#     # Step 2: Initialize an empty DataFrame to store the combined results
#     rh_df = pd.DataFrame()

#     # Step 3: Iterate over every 10 rows and append the results of 'query_data' to the 'result_df'
#     for i in range(0, total_rows, 10):
#         datapointids = df_meta_rh["datapointid"].iloc[i:i+10].values.tolist()
#         data = query_data(datapointids)
#         rh_df = pd.concat([rh_df, data])
#     rh_df['datetime'] = rh_df.index
#     rh_df.reset_index(drop=True, inplace=True)
#     rh_df['datapointid'] = rh_df['datapointid'].astype(int)


#     merged_df = rh_df.merge(df_meta[['datapointid', 'Room', 'Floor', 'FloorSection']], on='datapointid', how='left')

#     # Create the final dataframe with the desired structure
#     rh_df = pd.DataFrame({
#         'DATE': merged_df['datetime'],
#         'room': merged_df['Room'],
#         'indoor RH': merged_df['value'],
#     })


#     #OUTT
#     df_meta_outtemperature = df_meta[df_meta["category"] == "outdoor conditions: outdoor temperature, measured at location"].copy()
#     df_meta_outtemperature.reset_index(drop=True)
#     df_meta_outtemperature.dropna(subset=['Room'], inplace=True)
#     df_meta_outtemperature.drop(df_meta_outtemperature[df_meta_outtemperature.duplicated(subset='Room', keep=False)].index, inplace=True)
#     df_meta_outtemperature.reset_index(drop=True, inplace=True)
#     df_meta_outtemperature['datapointid'] = df_meta_outtemperature['datapointid'].astype(int)

#     # Get 10 first values from datapointid column to a list by getting 10 first
#     # rows and then getting the values from the column
#     total_rows = len(df_meta_outtemperature)

#     # Step 2: Initialize an empty DataFrame to store the combined results
#     outtemperature_df = pd.DataFrame()

#     # Step 3: Iterate over every 10 rows and append the results of 'query_data' to the 'result_df'
#     for i in range(0, total_rows, 10):
#         datapointids = df_meta_outtemperature["datapointid"].iloc[i:i+10].values.tolist()
#         data = query_data(datapointids)
#         outtemperature_df = pd.concat([outtemperature_df, data])

#     outtemperature_df['datetime'] = outtemperature_df.index
#     outtemperature_df.reset_index(drop=True, inplace=True)
#     outtemperature_df['datapointid'] = outtemperature_df['datapointid'].astype(int)
#     merged_df = outtemperature_df.merge(df_meta[['datapointid', 'Room', 'Floor', 'FloorSection']], on='datapointid', how='left')

#     # Create the final dataframe with the desired structure
#     outtemperature_df = pd.DataFrame({
#         'DATE': merged_df['datetime'],
#         'room': merged_df['Room'],
#         'floor': merged_df['Floor'],
#         'section': merged_df['FloorSection'],
#         'outT': merged_df['value'],
#     })

#     outtemperature_df=outtemperature_df.sort_values(by='DATE')
#     # outtemperature_df.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/fvh pilot/outdoor_FVH_260224.csv')

#     # df_new=[]

#     ####concat and create the merged dataset
#     result_df = pd.concat([co2_df, temperature_df, rh_df], ignore_index=True)
#     result_df_sorted = result_df.sort_values(by='DATE')
#     merged_df = pd.merge(temperature_df, rh_df, on=['DATE', 'room'])
#     merged_df = pd.merge(merged_df, co2_df, on=['DATE', 'room'])
#     # merged_df = pd.merge(merged_df, outtemperature_df, on=['DATE', 'room'])
#     merged_df=merged_df.sort_values(by=['DATE'])
#     merged_df=merged_df.reset_index(drop=True)


#     #aggiungere l'outdoor ai parametri indoor collected dalle room di interesse (25/33)
#     new_out=[]
#     room_label=merged_df['room'].unique()
#     dataallrooms=[]
#     for j in range(len(room_label)):
#         dataset_roomx=merged_df.loc[merged_df['room']==room_label[j]]
#         dataset_roomx=dataset_roomx.reset_index(drop=True)
#         dataset_roomx=dataset_roomx.sort_values(by='DATE')
#         #la temp outdoor è stata collected da 3 stanze diverse da quelle dell'indoor. Ho selezionato
#         #una delle tre stanze tanto le temp esterne sono uguali
#         out_T310=outtemperature_df.loc[outtemperature_df['room']==str(310)]
#         out_T310=out_T310.sort_values(by='DATE')
#         out_T310=out_T310.reset_index(drop=True)
#         #sincronizzazione delle date
#         #1) elimino dalle temp out tutti i campioni presi in date out fo range delle date degli indoor
#         enddate_ind=dataset_roomx['DATE'].max()
#         startdate_ind=dataset_roomx['DATE'].min()
#         enddate_out=out_T310['DATE'].max()
#         startdate_out=out_T310['DATE'].min()
#         #outdoor filtrato
#         filtered_outT = out_T310[(out_T310['DATE'] <= enddate_ind) | (out_T310['DATE'] >= startdate_ind)]
#         #elimino tutte quelle date che non sono in comune tra i due dataset (outdoor e indoor)
#         df_out=filtered_outT[filtered_outT.DATE.isin(dataset_roomx.DATE)]    
#         df2=dataset_roomx[dataset_roomx.DATE.isin(df_out.DATE)]
#         df_out=df_out.reset_index(drop=True)
#         df2=df2.reset_index(drop=True)
#         df2['outT']=df_out['outT']
#         dataallrooms.append(df2)
#         # new_out.append(df_out)

#     df3 = pd.concat(dataallrooms, ignore_index=True)
#     df3=df3.reset_index(drop=True)
#     # s4=str(file, 'utf-8')
#     # data4=StringIO(s4)
#     # df4=pd.read_csv(data4, usecols=selected_columns2)
#     # room_number = df['room'].tolist()
#     # df.drop('room', axis=1, inplace=True)
#     X4 = df3[['indoor T','indoor RH', 'co2', 'outT']]
#     lst3=X4.values.tolist()
#     unique_offices = df3['room'].unique()
#     results2 = {}

#     # data=environment.dict()
#     # data_in=[[data['indoorT'], data['indoorRH'],data['CO2'], data['outdoorT'],data['lux']]]
#     # print(data_in) 
    
#     model_name4='s3://mlflow/61/edae9d05e01d42f7906b31a43c2ff067/artifacts/LSTM24_univpm'
#     try:
#        # Load the MLflow model
#            if verbose:
#                print()
#            model4 = mlflow.sklearn.load_model(model_name4)
#            for j in range(len(unique_offices)-1):
#                room_data = df3[df3['room'] == unique_offices[j]][['indoor T', 'indoor RH', 'co2', 'outT']]

#                #divido gli input in train e test
#                train_size = int(len(room_data) * 0.8) 
#                X_train, X_test = room_data[:train_size], room_data[train_size:]
               
#                scaler = MinMaxScaler()
#                X_train_scaled = scaler.fit_transform(X_train)
#                X_test_scaled = scaler.fit_transform(X_test)
#                # Create sequences for LSTM model
#                sequence_length = 168  # 1 week (24h * 7)
#                forecast_horizon = 24  # Predicting the next 24 hours
#                # X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length, forecast_horizon)
#                # X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length, forecast_horizon)
#                X_train_seq = create_sequences(X_train_scaled, sequence_length, forecast_horizon)
#                X_test_seq = create_sequences(X_test_scaled, sequence_length, forecast_horizon)
#                room_comfort_pred = np.mean(model4.predict(X_test_seq))
#                results2[str(unique_offices[j])] = room_comfort_pred.tolist()
                   
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
 
#     return{ 'predicted comfort level': results2}





