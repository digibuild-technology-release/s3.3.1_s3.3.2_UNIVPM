import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
url_openemteo_forecast1 = os.getenv("URL_OPENMETEO_FORECAST1")
url_openemteo_forecast7 = os.getenv("URL_OPENMETEO_FORECAST7")
url_openemeteo_history = os.getenv("URL_WEATHER_HISTORY")

#&lat=60.17&lon=24.94 ----> HELSINKI
#basic-1h ----> libreria che  prende i dati foreast a un'ora

def get_weather_forecast_FVH():
    # Endpoint Open Meteo per 7 giorni
    url_7days = url_openemteo_forecast7
    response_7 = requests.get(url_7days)

    # Endpoint Open Meteo per 1 giorno
    url_1day = url_openemteo_forecast1
    response_1 = requests.get(url_1day)
    
    # Forecast 7 days
    json_data7 = response_7.json()
    print("7-day forecast", json_data7)
    temperature_data7 = json_data7["hourly"].get("temperature_2m", [])[0:168]  # Limita alle prime 168 ore (7 giorni)
    time7 = json_data7["hourly"].get("time", [])[0:168]
    df7 = pd.DataFrame()
    df7['date'] = time7
    df7['date'] = pd.to_datetime(df7['date'])
    df7['outT_next_7days'] = temperature_data7

    # Forecast 1 day
    json_data1 = response_1.json()
    temperature_data1 = json_data1["hourly"].get("temperature_2m", [])[24:-1]  # Limita alle ore per il giorno successivo
    time1 = json_data1["hourly"].get("time", [])[24:-1]
    df1 = pd.DataFrame()
    df1['date'] = time1
    df1['date'] = pd.to_datetime(df1['date'])
    df1['outT_next_1day'] = temperature_data1
    
    return df7, df1

def get_weather_hist_FVH():

    response_hist = requests.get(url_openemeteo_history)
    json_data_hist = response_hist.json()
    past_7_days = json_data_hist["hourly"].get("temperature_2m", [])[0:168]
    time_past_7days = json_data_hist["hourly"].get("time", [])[0:168]
    df_hist=pd.DataFrame()
    df_hist['date']=time_past_7days
    df_hist['date'] = pd.to_datetime(df_hist['date'], format="%Y-%m-%dT%H:%M")#modificato
    df_hist['outT_past_7days']=past_7_days
    
    return df_hist


####test the defined function
a, b= get_weather_forecast_FVH()
c=get_weather_hist_FVH()

