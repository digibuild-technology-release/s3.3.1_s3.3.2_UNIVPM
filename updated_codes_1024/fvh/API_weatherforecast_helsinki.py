
"""
Created on Tue Nov 30 16:26:42 2024

@author: vittoria
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

#&lat=60.17&lon=24.94 ----> HELSINKI
#basic-1h ----> libreria che  prende i dati foreast a un'ora

def get_weather_forecast_FVH():

    response_7 = requests.get('http://my.meteoblue.com/packages/basic-1h_basic-day_trend-1h_trend-day?&lat=60.17&lon=24.94&apikey=ldLzOIsYkoynJRXP&forecast_days=7')
    response_1 = requests.get('http://my.meteoblue.com/packages/basic-1h_basic-day_trend-1h_trend-day?&lat=60.17&lon=24.94&apikey=ldLzOIsYkoynJRXP&forecast_days=2')
    
    
    #forecast 7 days
    json_data7 = response_7.json()
    temperature_data7 = json_data7["data_1h"].get("temperature", [])
    time7 = json_data7["data_1h"].get("time", [])
    df7=pd.DataFrame()
    df7['date']=time7
    df7['date']=pd.to_datetime(df7['date'], format="%Y-%m-%d %H:%M")
    df7['outT_next']=temperature_data7
    
    #forecast 1 day
    json_data1 = response_1.json()
    temperature_data1 = json_data1["data_1h"].get("temperature", [])[24:-1]
    time1 = json_data1["data_1h"].get("time", [])[24:-1]
    df1=pd.DataFrame()
    df1['date']=time1
    df1['date']=pd.to_datetime(df1['date'], format="%Y-%m-%d %H:%M")
    df1['outT_next']=temperature_data1
    
    return df7, df1

####test the defined function
a=pd.DataFrame()
b=pd.DataFrame()

a, b= get_weather_forecast_FVH()

