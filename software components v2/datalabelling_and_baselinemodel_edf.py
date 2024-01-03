# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:25:50 2023

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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy import interpolate
from scipy.interpolate import interp1d
import mlflow
import time
import seaborn as sns

#%%%% Labeling 
####import the final dataset - FFS (the dataset has only February)
dataset_dir_FFS='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/edf_pilot/FFSouth_resampled.csv'
dataset_FFS=pd.read_csv(dataset_dir_FFS, sep=',')
dataset_FFS['DateTime']=pd.to_datetime(dataset_FFS['DateTime'],  format = "%Y-%m-%d %H:%M:%S")
####check the correlation between the input variables to select the valuable features -GF
correlation_matrix = dataset_FFS.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - First FLoor South ')
plt.show()
#sPMV
#sPMV calculation for all the year (from month 1 to month 12)
dataset_FFS['sensor']='id1'
spmv_FFS=sPMV_calculation(dataset_FFS['env_temp'], dataset_FFS['env_rh'],dataset_FFS['sensor'],dataset_FFS['DateTime']) # 
spmv_FFS=spmv_FFS.reset_index(drop=True)
X_FFS = dataset_FFS[['env_temp','env_rh', 'co2_co2', 'Temperature_out']] 
X_FFS['sPMV']=spmv_FFS['sPMV']


####import the final dataset - FFN (the dataset has data from feb to oct)
dataset_dir_FFN='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/edf_pilot/FFNorth_resampled.csv'
dataset_FFN=pd.read_csv(dataset_dir_FFN, sep=',')
dataset_FFN['DateTime']=pd.to_datetime(dataset_FFN['DateTime'],  format = "%Y-%m-%d %H:%M:%S")
####check the correlation between the input variables to select the valuable features - FFN
correlation_matrix = dataset_FFN.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - First FLoor North ')
plt.show()
#sPMV
#sPMV calculation for all the year (from month 1 to month 12)
dataset_FFN['sensor']='id2'
spmv_FFN=sPMV_calculation(dataset_FFN['env_temp'], dataset_FFN['env_rh'],dataset_FFN['sensor'],dataset_FFN['DateTime']) # 
spmv_FFN=spmv_FFN.reset_index(drop=True)
X_FFN = dataset_FFN[['env_temp','env_rh', 'co2_co2', 'Temperature_out', 'DateTime']] 
# X_FFN['sPMV']=spmv_FFN['sPMV']
# Extract unique dates from each dataset
unique_dates_df1 = set(dataset_FFN['DateTime'])
unique_dates_df2 = set(spmv_FFN['DATE'])
# Check if there are common dates
common_dates = unique_dates_df1.intersection(unique_dates_df2)
filtered_X_FFN = X_FFN[X_FFN['DateTime'].isin(common_dates)]
final_FFN=filtered_X_FFN[['env_temp','env_rh', 'co2_co2', 'Temperature_out']]#'DateTime',
# final_FFN['sPMV']=spmv_FFN['sPMV'].values
####save the dataset as csv
# final_FFN.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/edf_pilot/dataset_FFN.csv')


####import the final dataset - GF (the dataset has data from feb to june)
dataset_dir_GF='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/edf_pilot/GF_resampled.csv'
dataset_GF=pd.read_csv(dataset_dir_GF, sep=',')
dataset_GF['DateTime']=pd.to_datetime(dataset_GF['DateTime'],  format = "%Y-%m-%d %H:%M:%S")
####check the correlation between the input variables to select the valuable features -GF
correlation_matrix = dataset_GF.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - Ground Floor ')
plt.show()
# sPMV
#sPMV calculation for all the year (from month 1 to month 12)
dataset_GF['sensor']='ellona'
spmv_GF=sPMV_calculation(dataset_GF['env_temp'], dataset_GF['env_rh'],dataset_GF['sensor'],dataset_GF['DateTime']) # 
spmv_GF=spmv_GF.reset_index(drop=True)
X_GF = dataset_GF[['env_temp','env_rh', 'co2_co2', 'Temperature_out']][0:31968] #'DateTime',
# X_GF['sPMV']=spmv_GF['sPMV'].values
####save the dataset as csv
# X_GF.to_csv('C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/edf_pilot/dataset_GF.csv')


#%%%% train and test the model - data concerns the whole demoroom 2
####select the lable (spmv) from the dataset and drop the columns we dont need from the dataset
y=spmv_GF['sPMV']
# y=spmv_FFN['sPMV']

#sPMV must be categorized 
y_new=[]
for i in range(len(y)):
    if np.logical_and(y[i] >0.5, y[i] <=3.05)==True:
        y_new.append(1) 
    else: 
        if np.logical_and(y[i] >= -0.5, y[i] <= 0.5)==True:
            y_new.append(0)
        else: 
            if np.logical_and(y[i] >= -3, y[i] < -0.5)==True:
                y_new.append(-1)
                
           
X_train, X_test, y_train, y_test = train_test_split(X_GF, y_new, test_size=0.2, random_state=42)


####RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, f1_score, recall_score
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf = clf_rf.fit(X_train, y_train)
# clf_rf = RandomForestRegressor(n_estimators=100, random_state=42)
# model_rf = clf_rf.fit(X_train, y_train)
#Predict your test set on the trained model
my_output_rf = model_rf.predict(X_test)
cm_rf=confusion_matrix(y_test, my_output_rf)
accuracy_rf = accuracy_score(y_test, my_output_rf)
p_rf=precision_score(y_test, my_output_rf,average= 'macro')
f1_score_rf=f1_score(y_test, my_output_rf, average= 'macro')
recall_rf=recall_score(y_test, my_output_rf, average= 'macro')
plt.figure(figsize=(20,5))
plt.plot(y_test)
plt.plot(my_output_rf)  


#### DECISION TREE
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state=0)
#Predict your test set on the trained model
model_dt = clf_dt.fit(X_train, y_train)
my_output_dt = model_dt.predict(X_test)
cm_dt=confusion_matrix(y_test, my_output_dt)
accuracy_dt = accuracy_score(y_test, my_output_dt)
p_dt=precision_score(y_test, my_output_dt,average= "macro")
f1_score_dt=f1_score(y_test, my_output_dt, average= 'macro')
recall_dt=recall_score(y_test, my_output_dt, average= 'macro')

#### NB
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
model_NB = clf_nb.fit(X_train, y_train)
my_output_nb = model_NB.predict(X_test)
cm_nb=confusion_matrix(y_test, my_output_nb)
accuracy_nb = accuracy_score(y_test, my_output_nb)
p_nb=precision_score(y_test, my_output_nb,average= "macro")
f1_score_nb=f1_score(y_test, my_output_nb, average= 'macro')
recall_nb=recall_score(y_test, my_output_nb, average= 'macro')

#### KNN 
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier() #n_neighbors=4
model_knn = clf_knn.fit(X_train, y_train)
my_output_knn = model_knn.predict(X_test)
cm_knn=confusion_matrix(y_test, my_output_knn)
accuracy_knn = accuracy_score(y_test, my_output_knn)
p_knn=precision_score(y_test, my_output_knn,average= "macro")
f1_score_knn=f1_score(y_test, my_output_knn, average= 'macro')
recall_knn=recall_score(y_test, my_output_knn, average= 'macro')

#### ADABOOST
from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier(n_estimators=100, random_state=0)
model_ada = clf_ada.fit(X_train, y_train)
my_output_ada = model_ada.predict(X_test)
cm_ada=confusion_matrix(y_test, my_output_ada)
accuracy_ada = accuracy_score(y_test, my_output_ada)
p_ada=precision_score(y_test, my_output_ada,average= "macro")
f1_score_ada=f1_score(y_test, my_output_ada, average= 'macro')
recall_ada=recall_score(y_test, my_output_ada, average= 'macro')

#### BAGGING
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
clf_bag = BaggingClassifier() #base_estimator=SVC(kernel = 'linear'),n_estimators=8, random_state=0
model_bag = clf_bag.fit(X_train, y_train)
my_output_bag = model_bag.predict(X_test)
cm_bag=confusion_matrix(y_test, my_output_bag)
accuracy_bag = accuracy_score(y_test, my_output_bag)
p_bag=precision_score(y_test, my_output_bag,average= "macro")
f1_score_bag=f1_score(y_test, my_output_bag, average= 'macro')
recall_bag=recall_score(y_test, my_output_bag, average= 'macro')
# a_score_bag=accuracy_score(y_test, my_output_bag,normalize=True)
# mlflow.sklearn.log_model(rf, "model", signature=signature)

from tabulate import tabulate
table_data = [
    ["algorithm", "accuracy", "f1_score", "precision", "recall"],
    ["RF", accuracy_rf, f1_score_rf, p_rf,recall_rf],
    ["DT", accuracy_dt, f1_score_dt, p_dt,recall_dt],
    ["NB", accuracy_nb, f1_score_nb, p_nb,recall_nb],
    ["KNN", accuracy_knn, f1_score_knn, p_knn,recall_knn],
    ["ADA", accuracy_ada, f1_score_ada, p_ada,recall_ada],
    ["BAG", accuracy_bag, f1_score_bag, p_bag,recall_bag]
]

# Print the table
print(tabulate(table_data, headers=['Column 1', 'Column 2', 'Column 3', 'Column 4', 'Column 5'], tablefmt='grid'))


