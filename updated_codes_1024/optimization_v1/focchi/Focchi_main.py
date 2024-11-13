# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:23:35 2024

@author: utente
"""

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from FOCCHI_functions_opt_paretoefficient import *    
import pandas as pd

INPUT_CSV_FILE = 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/OTTIMIZZAZIONE_COLLAB/per marinelli/datasets/focchi/input_focchi.csv'                                              
df_or=pd.read_csv(INPUT_CSV_FILE, sep=';')

h_start_work=9
h_stop_work=18
df_d, df_n=split_day_night_focchi(df_or, h_start_work, h_stop_work)


df=df_d.loc[0:len(df_d)-10] 

NEXT_TEMP=tuple(df_d['outT'][len(df):len(df)+10]) #to be changed with the outdoor temperature of the next day during working hours (9:00-18:00)
K =10
DIST_SOGLIA= 2 #(outTmax-outTmin)np.round((max(NEXT_TEMP)- min(NEXT_TEMP)), 2)  #

lista = leggi_e_memorizza_csv(df, False)
records = unisci_tuple(lista, K)
records = compute_temperature_dist(records,NEXT_TEMP)

best_records = sorted(
    (tupla for tupla in records if tupla[DIST] <= DIST_SOGLIA),
    key=lambda x: x[DIST]  
)

best_records = getParetoSet(best_records)
    
res=plotParetoSet(best_records, records, lista, K)  
first_key, first_value = next(iter(res.items()))
output=first_value[2]
print(output)



