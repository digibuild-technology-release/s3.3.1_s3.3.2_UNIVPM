# -*- coding: utf-8 -*-
"""
FUNCTION FOR THE sPMV CALCULATION ACCORDING TO BURATTI ET AL.

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

#%%%% sPMV FUNCTION: this funtion compute the sPMV accorign to the month you give as inout. Indeed, the sPMV equation changes accoring to the Iclo, hence to the season.
#%%%% inputs: (1) indoor T (dataframe), (2)indoor RH (dataframe), date (series- datetime), sensor (series), and months( series). These inputs are taken from the created input dataset.

def sPMV_calculation(indoorT, indoorRH, sensor, date,  months):
    #calculation of vapor pressuresaturation - magnus formula
    # months=indoorT['DATE'].dt.month
    indoorES = pd.Series()   
    indoorES = 0.61094 * math.e**((17.625 * (indoorT)) / ((indoorT) + 243.04))
    # Calculate water vapor pressure (pv)
    pv=pd.DataFrame()
    pv['vapour_pressure'] = indoorES*( indoorRH  / 100)
    pv['DATE']= date
   
    for i in months:
        ####if you select 0 as month --> reference model will be implemented
        if i==0:
            # simplified PMV with coefficient by Rohles - these coefficients are calculated considering a Iclo=0.6 (=trousers, long-sleeved shirt)
            a=0.245
            b=0.248
            c=6.475
            
            pmv=round((a*indoorT+b*pv['vapour_pressure']-c),2)
            sPMV=pd.DataFrame()
            sPMV['DATE']=date
            sPMV['sPMV']=pmv
            sPMV['related n_sensor']=sensor
            sPMV['applied_model']='gound truth - simplified pmv'
            # print('-------selected model rholes - tot--------')
            # print(sPMV['DATE'])
            # print(sPMV['sPMV'])
            # print(a)
            # print(b)
            # print(c)
           
            return sPMV
    ####if you select from 1 to 3 as month --> buratti et al. model will be implemented for winter season
    #winter
        if i >= 1 and i <= 3:
            #simplified PMV with coefficient by Buratti et al - these coefficients have been computed by considering different Iclo
            #in this case, Iclo[1.01-1.65]
            a_c=0.1478
            b_c=-0.1371
            c_c=2.5239
            pv=pv.loc[(date.dt.month== months)]
            pmv_w=round((a_c*(indoorT.loc[(date.dt.month== months)])+b_c*pv['vapour_pressure']-c_c), 2)
            sPMV_w=pd.DataFrame()
            sPMV_w['DATE']=date.loc[(date.dt.month== months)]
            sPMV_w['sPMV']=pmv_w.astype(float)
            sPMV_w['related n_sensor']=sensor.loc[(date.dt.month== months)]
            sPMV_w['applied_model']='simplified pmv - w'
            
            # print('-------selected model buratti - w--------')
            # print(sPMV_w['DATE'])
            # print(sPMV_w['sPMV'])
            # print(a_c)
            # print(b_c)
            # print(c_c)
            
            return sPMV_w
    
        else:
        
            if i==12:
                #simplified PMV with coefficient by Buratti et al - these coefficients have been computed by considering different Iclo
                #in this case, Iclo[1.01-1.65]
                a_c=0.1478
                b_c=-0.1371
                c_c=2.5239
                pv=pv.loc[(date.dt.month== months)]
                pmv_w=round((a_c*(indoorT.loc[(date.dt.month== months)])+b_c*pv['vapour_pressure']-c_c), 2)
                sPMV_w=pd.DataFrame()
                sPMV_w['DATE']=date.loc[(date.dt.month== months)]
                sPMV_w['sPMV']=pmv_w.astype(float)
                sPMV_w['related n_sensor']=sensor.loc[(date.dt.month== months)]
                sPMV_w['applied_model']='simplified pmv - w'
                
                # print('-------selected model buratti - w--------')
                # # print(sPMV['applied_model'])
                # print(sPMV_w['DATE'])
                # print(sPMV_w['sPMV'])
                # print(a_c)
                # print(b_c)
                # print(c_c)
                
                return sPMV_w
            else:
                print('outofrange')
        
    ####if you select from 4 to 5 and from 9 to 11 as month --> buratti et al. model will be implemented for middle seasons
    #middle season
        if i >= 4 and i <= 5 :
            #simplified PMV with coefficient by Buratti et al - these coefficients have been computed by considering different Iclo
            #in this case, Iclo[0.51-1.00]
            a_b=0.1383
            b_b=0.0269
            c_b=3.0190
            pv=pv.loc[(date.dt.month== months)]
            pmv_m=round((a_b*indoorT.loc[(date.dt.month== months)]+b_b*pv['vapour_pressure']-c_b), 2)
            sPMV_m=pd.DataFrame()
            sPMV_m['DATE']=date.loc[(date.dt.month== i)]
            sPMV_m['sPMV']=pmv_m.astype(float)
            sPMV_m['related n_sensor']=sensor.loc[(date.dt.month== months)]
            sPMV_m['applied_model']='simplified pmv - m'
            
            # print('-------selected model buratti - m--------')
            # # print(sPMV['applied_model'])
            # print(sPMV_m['DATE'])
            # print(sPMV_m['sPMV'])
            # print(a_b)
            # print(b_b)
            # print(c_b)
            
            return sPMV_m
        else:
            if i>=9 and i <= 11:
                #simplified PMV with coefficient by Buratti et al - these coefficients have been computed by considering different Iclo
                #in this case, Iclo[0.51-1.00]
                a_b=0.1383
                b_b=0.0269
                c_b=3.0190
                pv=pv.loc[(date.dt.month== months)]
                pmv_m=round((a_b*indoorT.loc[(date.dt.month== months)]+b_b*pv['vapour_pressure']-c_b), 2)
                sPMV_m=pd.DataFrame()
                sPMV_m['DATE']=date.loc[(date.dt.month== months)]
                sPMV_m['sPMV']=pmv_m.astype(float)
                sPMV_m['related n_sensor']=sensor.loc[(date.dt.month==months)]
                sPMV_m['applied_model']='simplified pmv - m'
                # print('-------selected model buratti - m--------')
                # print(sPMV_m['DATE'])
                # print(sPMV_m['sPMV'])
                # print(a_b)
                # print(b_b)
                # print(c_b)
                return sPMV_m
            
            else: 
                    
                print('outofrange')
                
        ####if you select from 8 to 6 as month (WARNING: data from june have been not provided) --> buratti et al. model will be implemented for summer season
        #summer
        if i >= 7 and i <= 8   :
            #simplified PMV with coefficient by Buratti et al - these coefficients have been computed by considering different Iclo
            #in this case, Iclo[025-0.50]
            a_a=0.2803
            b_a=0.1717
            c_a=7.1383
            pv=pv.loc[(date.dt.month== months)]
            # indoorT=29.02
            # pv=28.1656
            pmv_s=round((a_a*(indoorT.loc[(date.dt.month== months)])+b_a*(pv['vapour_pressure'])-c_a), 2)
            sPMV_s=pd.DataFrame()
            sPMV_s['DATE']=date.loc[(date.dt.month== months)]
            sPMV_s['sPMV']=pmv_s.astype(float)
            sPMV_s['related n_sensor']=sensor.loc[(date.dt.month==months)]
            sPMV_s['applied_model']='simplified pmv - s'
            # print('-------selected model buratti - s--------')
            # # print(sPMV['applied_model'])
            # print(sPMV_s['DATE'])
            # print(sPMV_s['sPMV'])
            # print(a_a)
            # print(b_a)
            # print(c_a)
            return sPMV_s
        else:
                print('outofrange')

  




                
