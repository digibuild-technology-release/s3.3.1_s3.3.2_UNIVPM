# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:07:43 2024

@author: vittoria cipollone

--------------------------------------------------------------------------------------------

PYTHON FUNCTION FOR COMFORT ADAPTIVE MODEL by NORM 1525100

This funztion takes as input: 
    1) indoor temperatures ---> series of float
    2) outdoor temperatures --->series of  float
    3) dates ---> series of datetime
    
The iutput of the function will be a dataframe with the following columns: 
    1) date ---> datetime
    2) upper comfort limit (operative temperature) ---> float
    3) lower comfort limit (operative temperature) ---> float
    4) status, i.e., a string indicating comfort or discomfort situation ---> str
    5) bin, i.e., 0 or 1 associated to comfort and discomfort rispectively ---> int

--------------------------------------------------------------------------------------------
"""

import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
import requests
from datetime import datetime

# Tin, Tout, date
def adaptive_model(Tin, Tout, date):
        
        #calcolo temperatura media outdoor (tr) di ultimi 7 giorno
        running_means = []
        weights = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        denominator = 3.8 #sum(weights)
        for i in range(6,len(Tout)):
            weighted_sum = 0
            for j in range(7):
                weighted_sum += weights[j] * Tout[i-j]
            running_means.append(weighted_sum / denominator)
        tr=np.array(running_means)
        #upper and lower comfort limit (NORM 1525100)
        upper_c=0.33*tr + 18.8 + 3 
        lower_c=0.33*tr + 18.8 - 3     
                  
        #nei due df denominati output 1 e 2 si ha: 
            # status: 'comf', 'disc' ---> str
            # bin: 0 (per 'comf'), 1 (per 'disc') ---> int
            # date ---> datetime 
            # upperL, lowerL: temperature operative (da normativa) ----> float 
        
        output1 = pd.DataFrame(columns=['date','upperL', 'lowerL', 'status','bin'])
        output2 = pd.DataFrame(columns=['date', 'upperL', 'lowerL','status', 'bin'])
        Tin = Tin[0:len(tr)]
        date = date[0:len(tr)]

        for k in range(len(tr)):
        # if df[(df['tin'] > df_c_limits['upper_c']) | (df['tin'] < df_c_limits['lower_c'])]:
            if Tin[k] >= upper_c[k] or Tin[k] <= lower_c[k]:
                # report.append(comf_disc[1])
                output1 = output1.append({'date': date[k], 'Tin':Tin[k],'upperL':upper_c[k], 'lowerL': lower_c[k] ,'status': 'disc', 'bin':1, }, ignore_index=True)
        
            else: 
                output2 = output2.append({'date': date[k], 'Tin':Tin[k],'upperL':upper_c[k], 'lowerL': lower_c[k],'status': 'comf', 'bin':0}, ignore_index=True)
        
        #join i due df con output del modello adattivo 
        merged_df = pd.concat([output1, output2], ignore_index=True)
        merged_df_sort=merged_df.sort_values(by='date')
        merged_df_sort=merged_df_sort.reset_index(drop=True)
        #return singolo valore del dataframe
        
        
        return merged_df_sort
    
