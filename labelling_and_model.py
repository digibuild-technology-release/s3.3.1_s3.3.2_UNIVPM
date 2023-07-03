# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:25:50 2023

@author: utente
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

#%%%% Labeling 
####import the final dataset
dataset_dir='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/input dataset/focchi_input_dataset_univpm.csv'

dataset=pd.read_csv(dataset_dir, sep=',')
dataset=dataset.drop(['Unnamed: 0'], axis=1)

# Temperature in degrees Celsius
# dataset_new=dataset.drop((dataset[(dataset['data_type']=='indoor T')&(dataset['month']==8)&(dataset['day']==1)].index))
indoorT=pd.DataFrame()
indoorT['indoor T']= dataset['values'].loc[(dataset['data_type']== 'indoor T')]
indoorT['n_sensor']= dataset['n_sensor'].loc[(dataset['data_type']== 'indoor T')]
indoorT['DATE']= dataset['DATE'].loc[dataset['data_type']== 'indoor T']
indoorT=indoorT.reset_index(drop=True)

# Relative humidity in percentage
indoorRH=pd.DataFrame()
indoorRH['ind RH'] = dataset['values'].loc[dataset['data_type']== 'indoor RH']
indoorRH['n_sensor']= dataset['n_sensor'].loc[(dataset['data_type']== 'indoor RH')]
indoorRH['DATE']= dataset['DATE'].loc[dataset['data_type']== 'indoor RH']
indoorRH=indoorRH.reset_index(drop=True)

#remove the nan values from both rh and t
index_RH_nan = indoorRH[indoorRH.isna().any(axis=1)].index
index_T_nan = indoorT[indoorT.isna().any(axis=1)].index
indoorT=indoorT.drop(index_RH_nan)
indoorRH=indoorRH.drop(index_T_nan)
indoorT=indoorT.dropna()
indoorRH=indoorRH.dropna()
indoorT=indoorT.reset_index(drop=True)
indoorRH=indoorRH.reset_index(drop=True)

####simplified pmv calculation 
indoorES=[]

for k in range(len(indoorRH)):
    #convert into datetime
    indoorT['DATE'][k]=datetime.strptime(indoorT['DATE'][k], "%Y-%m-%d %H:%M:%S")
    indoorRH['DATE'][k]=datetime.strptime(indoorRH['DATE'][k], "%Y-%m-%d %H:%M:%S")   
    # Calculate saturation vapor pressure using the Magnus formula
    es = 6.112 * math.exp((17.67 * indoorT['indoor T'][k]) / (indoorT['indoor T'][k] + 243.5))
    indoorES.append(es)
    
# Calculate water vapor pressure (pv)
pv=pd.DataFrame()
pv['vapur_pressure'] = indoorRH['ind RH'] * indoorES / 100
pv['DATE']= indoorRH['DATE']

# simplified PMV with coefficient by Rohles - these coefficients are calculated considering a Iclo=0.6 (=trousers, long-sleeved shirt)
a=0.245
b=0.248
c=6.475
pmv=round((a*indoorT['indoor T']+b*pv['vapur_pressure']-c),2)
sPMV=pd.DataFrame()
sPMV['DATE']=indoorRH['DATE']
sPMV['sPMV']=pmv

#simplified PMV with coefficient by Buratti et al - these coefficients have been computed by considering different Iclo
#in this case I choose coefficients that consider a Iclo[0.51-1.00]
a_b=0.1383
b_b=0.0269
c_b=3.0190

pmv_b=round((a_b*indoorT['indoor T']+b_b*pv['vapur_pressure']-c_b), 2)
sPMV_b=pd.DataFrame()
sPMV_b['DATE']=indoorRH['DATE']
sPMV_b['sPMV']=pmv_b.astype(float)
sPMV_b['related n_sensor']=indoorRH['n_sensor']

####selection of the paramters related to one zone of the whole demoroom and related to a specific period (spring: april-june)
# indoorT1=indoorT[((pd.to_datetime(indoorT['DATE'])).dt.month>=4)&((pd.to_datetime(indoorT['DATE'])).dt.month<=6)]
# indoorT1=indoorT1[(indoorT1['n_sensor']>=1) & (indoorT1['n_sensor']<=8)]
# indoorRH1=indoorRH[((pd.to_datetime(indoorRH['DATE'])).dt.month>=4)&((pd.to_datetime(indoorRH['DATE'])).dt.month<=6)]
# indoorRH1=indoorRH1[(indoorRH1['n_sensor']>=1) & (indoorRH1['n_sensor']<=8)]
# spmv1=sPMV_b[((pd.to_datetime(sPMV_b['DATE'])).dt.month>=4)&((pd.to_datetime(sPMV_b['DATE'])).dt.month<=6)]
# spmv1=spmv1[(spmv1['related n_sensor']>=1) & (spmv1['related n_sensor']<=8)]

# ####plots
# n=range(7)
# for a in n:
            
#                 temp = indoorT1['indoor T'].loc[(indoorT1['n_sensor']==a+1)]
#                 rh = indoorRH1['ind RH'].loc[(indoorRH1['n_sensor']==a+1)]
#                 date=    indoorT1['DATE'].loc[(indoorT1['n_sensor']==a+1)]  
#                 pmv=spmv1['sPMV'].loc[(spmv1['related n_sensor']==a+1)]
                
#                 # print(ds_giorno)
#                 # plt.figure(figsize=(20,10))
#                 # plt.plot(date, temp) #ogni colore è un giorno diverso ['indoor T'].values
#                 # plt.xlabel('date')
#                 # plt.ylabel('temperatures (°C)')
#                 # plt.plot(date, rh) #ogni colore è un giorno diverso ,temp['ind RH'].values
#                 # plt.xlabel('date')
#                 # plt.ylabel('RH (%)')
#                 # plt.title('indoor T and RH from April to June')
#                 fig, axs = plt.subplots(2, 1, figsize=(20, 10))

#                 # Plot per il primo subplot
#                 axs[0].plot(date, temp)
#                 axs[0].plot(date, rh, color='r')
#                 axs[0].set_title('indoor T from April to June')
#                 axs[0].set_xlabel('date')
#                 axs[0].legend('Indoor T (°C)', 'Indoor RH (%)')

#                 # Plot per il secondo subplot
#                 # axs[1].plot(date, rh)
#                 # axs[1].set_title('RH from April to June')
#                 # axs[1].set_xlabel('date')
#                 # axs[1].set_ylabel('Indoor RH (%)')

#                 axs[1].bar(date, pmv)
#                 axs[1].set_title('simplified PMV (b)')
#                 axs[1].set_xlabel('date')
#                 axs[1].set_ylabel('sPMV')
                
#%%%% TRIAL1: train and test the model
X=pd.DataFrame()
X['indoor T']=indoorT['indoor T']
X['indoor RH']=indoorRH['ind RH']
y=sPMV_b['sPMV'].dropna()

#sPMV must be categorized 
y_new=[]
for i in range(len(y)):
    if y[i] > 0.5 and y[i] <= 3:
        y_new.append(1) 
    else: 
        if np.logical_and(y[i] >= -0.5, y[i] <= 0.5)==True:
            y_new.append(0)
        else: 
            if np.logical_and(y[i] >= -3, y[i] < 0.5)==True:
                y_new.append(-1)
                

X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size=0.2, random_state=42)

####RANDOM FOREST
clf = RandomForestClassifier(max_depth=2, random_state=0)
model = clf.fit(X_train, y_train)
#Predict your test set on the trained model
my_output = model.predict(X_test)
# cm_rf=confusion_matrix(y_test, my_output)
accuracy_rf = accuracy_score(y_test, my_output)

####ML flow
# import mlflow
# import mlflow
# from mlflow.models.signature import infer_signature
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_diabetes
# from sklearn.ensemble import RandomForestRegressor

# with mlflow.start_run() as run:

#     db = load_diabetes()
#     X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
    
#     # Create and train models.
#     rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
#     rf.fit(X_train, y_train)
    
#     # Use the model to make predictions on the test dataset.
#     predictions = rf.predict(X_test)
    
#     signature = infer_signature(X_test, predictions)
#     mlflow.sklearn.log_model(rf, "model", signature=signature)

#     print("Run ID: {}".format(run.info.run_id))

# #%%%% export model in ML flow
# import mlflow

# # Start an MLflow run
# mlflow.start_run()

# # Log your model parameters and metrics
# mlflow.log_param('param1', value1)
# mlflow.log_metric('metric1', value1)

# # Save your model as an artifact
# mlflow.sklearn.save_model(model, 'model')

# # Set the HTTP endpoint for the model
# mlflow.set_tag('mlflow.deploy.http.endpoint', 'http://your-specific-endpoint')

# # End the MLflow run
# mlflow.end_run()



