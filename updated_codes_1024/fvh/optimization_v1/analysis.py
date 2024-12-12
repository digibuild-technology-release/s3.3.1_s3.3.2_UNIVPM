# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:37:01 2024

@author: utente
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle

file_path1 = "C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/OTTIMIZZAZIONE_COLLAB/per marinelli/datasets/fvh/processed_fvh_ntua_TSVml_room117.csv"
file_path2 = "C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/OTTIMIZZAZIONE_COLLAB/per marinelli/datasets/fvh/processed_fvh_ntua_TSVml_room121.csv"
file_path3 = "C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/OTTIMIZZAZIONE_COLLAB/per marinelli/datasets/fvh/processed_fvh_ntua_TSVml_room326.csv"
file_path4 = "C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/OTTIMIZZAZIONE_COLLAB/per marinelli/datasets/fvh/processed_fvh_ntua_TSVml_room333.csv"

#upload df
df = pd.read_csv(file_path4, sep=';')
df['DATE']=pd.to_datetime(df['DATE'])
df=df.drop(columns=['Unnamed: 0'])

# Resample to hourly, calculating the mean value for each hour
df['DATE']=pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)  # Set the timestamp column as index
df = df.resample('H').mean().interpolate()
df=df.reset_index()


#first check nan - before filtering
columns_with_nan = df.columns[df.isnull().any()].tolist()
print(f"Check 1: Colonne con NaN: {columns_with_nan}")
rows_with_nan = df[df.isnull().any(axis=1)]
print("Check 1: Righe con NaN:")
print(rows_with_nan)

#prefiltraggio
plt.figure(figsize=(15,10))
plt.plot(df['DATE'], df['value'])
plt.xlabel('dates')
plt.ylabel('energy cons')

#filtraggio outliers
mean_value = df['value'].mean().round(2)
std_dev = df['value'].std().round(2)
k=2
# Definizione dei limiti
lower_limit = mean_value - k*std_dev
upper_limit = mean_value + k*std_dev

df['value_outlier'] = (df['value'] < lower_limit) | (df['value'] > upper_limit)
df['value_rolling'] = df['value']
df.loc[df['value_outlier'], 'value_rolling'] = None
df['value_rolling'] = df['value_rolling'].fillna(df['value_rolling'].rolling(window=5, min_periods=1).mean()) #.interpolate(method='linear') #
df['value_rolling'] = df['value_rolling'].interpolate(method='linear', limit_direction='both')
df['value']=df['value_rolling'] 

#check 2 per Nan - post filtragio
columns_with_nan = df.columns[df.isnull().any()].tolist()
print(f"Check 2: Colonne con NaN: {columns_with_nan}")
rows_with_nan = df[df.isnull().any(axis=1)]
print("Check 2: Righe con NaN:")
print(rows_with_nan)


plt.figure(figsize=(15,10))
plt.plot(df['DATE'], df['value'])
# plt.plot(df['date'], df['value_rolling'])

plt.xlabel('dates')
plt.ylabel('energy cons')

df=df.drop(columns=['value_outlier', 'value_rolling', 'TSV_room_ml'])
df=df.reset_index(drop=True)

from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
from sPMV_v1 import *
# tsv_model = pickle.load(open('prova_random_forest.pkl', 'rb'))

tsv_model = pickle.load(open('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/Attivita_Vittoria/PAPERS/paper_sPMV/python/2_case_study_FVH/metodo_RFR_FVH.pkl', 'rb')) #'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/models/random_forest.pkl'
X_now = pd.DataFrame()
X_now['indoor T_y'] = df['indoor T']
X_now['indoor RH_y'] = df['indoor RH']
X_now['outT'] = df['outT']

tsv_now = tsv_model.predict(X_now)
tsv_now=np.round(tsv_now, 2)

df['tsv']=tsv_now

sPMV= sPMV_calculation(df['indoor T'], df['indoor RH'], df['DATE']).sort_values(by='DATE').reset_index(drop=True)

df['spmv']=sPMV['sPMV']#.reset_index(drop=True)

df.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/OTTIMIZZAZIONE_COLLAB/per marinelli/fvh/dataset91224/processed_fvh_ntua_room333_91224.csv')
