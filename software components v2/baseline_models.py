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
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
#%%%% import dataset 
dataset_dir='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/focchi_pilot/data_focchi_by_univpm/input dataset/focchi_input_dataset_61223_allrooms.csv'
dataset=pd.read_csv(dataset_dir, sep=',')
dataset=dataset.drop(['Unnamed: 0'], axis=1)
dataset['date']=pd.to_datetime(dataset['date'], format = "%Y-%m-%d %H:%M:%S")

#%%%% train and test the model 

####select the lable (spmv) from the dataset and drop the columns we dont need from the dataset
y=dataset['sPMV']
Xml=dataset.drop(['date','sPMV', 'room'], axis=1)

#sPMV must be categorized 
y_new=[]
for i in range(len(y)):
    if np.logical_and(y[i] >0.5, y[i] <=3)==True:
        y_new.append(1) 
    else: 
        if np.logical_and(y[i] >= -0.5, y[i] <= 0.5)==True:
            y_new.append(0)
        else: 
            if np.logical_and(y[i] >= -3, y[i] < -0.5)==True:
                y_new.append(-1)

X_train, X_test, y_train, y_test = train_test_split(Xml, y_new, test_size=0.2, random_state=42)

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
qwerty

#%%%% ML flow
import os
os.environ["MLFLOW_TRACKING_URI"]= "http://digibuild.epu.ntua.gr:5000/"
os.environ["AWS_SECRET_ACCESS_KEY"]= "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"]= "http://digibuild.epu.ntua.gr:9000/"
os.environ["AWS_ACCESS_KEY_ID"]= "minio"

# mlflow.set_tracking_uri("http://digibuild.epu.ntua.gr:5000")
# define model input features and signature
input_schema = Schema([
  ColSpec("float", "indoor T"),
  ColSpec("float", "indoor RH"),
  ColSpec("float", "CO2"),
  ColSpec("float", "outdoor T"),
  ColSpec("float", "lux")
])

output_schema = Schema([ColSpec("long", "sPMV")])

# signature consists of model input and output schemas
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
#experiment creation
experiment_name = "FOCCHI_s331_baseline_v2"
model_name1 = "RF_univpm"
model_name2= "DT_univpm"
model_name3= "NB_univpm"
model_name4= "KNN_univpm"
model_name5= "ADA_univpm"
model_name6= "BAG_univpm"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    
    # mlflow log model
    mlflow.sklearn.log_model(clf_rf, model_name1, signature=signature, registered_model_name=model_name1)
    # mlflow log params
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy_score(y_test, my_output_rf))
    mlflow.log_metric("precision", precision_score(y_test, my_output_rf,average= 'macro'))
    mlflow.log_metric("F1", f1_score(y_test, my_output_rf, average= 'macro'))
    mlflow.log_metric("recall", recall_score(y_test, my_output_rf, average= 'macro'))
    
with mlflow.start_run():
    mlflow.sklearn.log_model(clf_dt, model_name2, signature=signature, registered_model_name=model_name2)
    # mlflow log params
    mlflow.log_param("random_state", 0)
    mlflow.log_metric("accuracy", accuracy_score(y_test, my_output_dt))
    mlflow.log_metric("precision", precision_score(y_test, my_output_dt,average= 'macro'))
    mlflow.log_metric("F1", f1_score(y_test, my_output_dt, average= 'macro'))
    mlflow.log_metric("recall", recall_score(y_test, my_output_dt, average= 'macro'))
    
with mlflow.start_run():
    mlflow.sklearn.log_model(clf_nb, model_name3, signature=signature, registered_model_name=model_name3)
    # mlflow log params
    mlflow.log_param("random_state", 0)
    mlflow.log_metric("accuracy", accuracy_score(y_test, my_output_nb))
    mlflow.log_metric("precision", precision_score(y_test, my_output_nb,average= 'macro'))
    mlflow.log_metric("F1", f1_score(y_test, my_output_nb, average= 'macro'))
    mlflow.log_metric("recall", recall_score(y_test, my_output_nb, average= 'macro'))
    
with mlflow.start_run():
    mlflow.sklearn.log_model(clf_knn, model_name4, signature=signature, registered_model_name=model_name4)
    # mlflow log params
    mlflow.log_param("n_neighbors", 5)
    mlflow.log_param("weights", 'uniform')
    mlflow.log_param("leaf_size", 30)
    mlflow.log_metric("accuracy", accuracy_score(y_test, my_output_knn))
    mlflow.log_metric("precision", precision_score(y_test, my_output_knn,average= 'macro'))
    mlflow.log_metric("F1", f1_score(y_test, my_output_knn, average= 'macro'))
    mlflow.log_metric("recall", recall_score(y_test, my_output_knn, average= 'macro'))
    
with mlflow.start_run():
    mlflow.sklearn.log_model(clf_ada, model_name5, signature=signature, registered_model_name=model_name5)
    # mlflow log params
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 0)
    mlflow.log_metric("accuracy", accuracy_score(y_test, my_output_ada))
    mlflow.log_metric("precision", precision_score(y_test, my_output_ada,average= 'macro'))
    mlflow.log_metric("F1", f1_score(y_test, my_output_ada, average= 'macro'))
    mlflow.log_metric("recall", recall_score(y_test, my_output_ada, average= 'macro'))

with mlflow.start_run():
    mlflow.sklearn.log_model(clf_bag, model_name6, signature=signature, registered_model_name=model_name6)
    # mlflow log params
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("random_state", 'None')
    mlflow.log_param("estimator", 'DecisionTreeClassifier')
    mlflow.log_metric("accuracy", accuracy_score(y_test, my_output_bag))
    mlflow.log_metric("precision", precision_score(y_test, my_output_bag,average= 'macro'))
    mlflow.log_metric("F1", f1_score(y_test, my_output_bag, average= 'macro'))
    mlflow.log_metric("recall", recall_score(y_test, my_output_bag, average= 'macro'))    
