# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:23:35 2024

@author: utente
"""
from ucl_paretofunction import *
# from adaptive_function import adaptive_model
from api_client_auth import *
import pandas as pd


INPUT_CSV_FILE = 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/ucl_opt_v0/datasets/020_processed_cut_new.csv' 

df_or=pd.read_csv(INPUT_CSV_FILE, sep=',')
df_or['date']=pd.to_datetime(df_or['date'])
df_or=df_or.round(2)

unnamed_columns = [col for col in df_or.columns if col.startswith('Unnamed')]
if unnamed_columns:
    print(f"Dropping columns: {unnamed_columns}")
    df_or.drop(columns=unnamed_columns, inplace=True)
else:
    print("No 'Unnamed' columns found. Proceeding without changes.")
    

df_work=[]
df_night=[]
h_start_work=9
h_stop_work=18

df_work, df_night=split_day_night(df_or, h_start_work, h_stop_work) 

K =10 
df_to_use=df_work[0:len(df_work)-10].reset_index(drop=True)
df_next_day=df_work[len(df_work)-10: len(df_work)].reset_index(drop=True)
#weather forecast from BBI     
NEXT_TEMP = tuple(round(df_next_day['outT'],2))   
# NEXT_TEMP=(5.78, 5.55, 5.65, 5.9, 5.98, 5.83, 5.37, 4.97, 4.25, 3.9)        
#API by CARTIF for energy prediction of the next day
username = 'vcipollone'
password = '4K@y0<u6kVD('

token = gen_token(username, password)
pilot_name1 = 'UCL'
variable = 'energy_consumption'
frequency = 'hourly'
model_retrieved1 = get_models(token, pilot_name1)
info_retrieved1 = get_info(token, pilot_name1, variable, frequency)
predictions = get_prediction(token, pilot_name1, variable, frequency)
room = "020"  #here 020 should be changed with the id of the current room
valori = list(filter(lambda d: room in d, predictions)) 
ENERGY_CONS = [d[room] for d in valori][h_start_work:h_stop_work+1]
ENERGY_CONS =  tuple(map(lambda x: round(x, 2), ENERGY_CONS))#
                                         
DIST_TEMP_SOGLIA = 0.7                                              
DIST_SPMV_SOGLIA = 0.7                                             

lista = leggi_e_memorizza_csv(df_to_use.round(2), False)

records = unisci_tuple(lista, K)

records = compute_temperature_dist(records, ENERGY_CONS,NEXT_TEMP, K)


print_dataset(records, K, True)


SPMV_AS_KPI = False #false = spmv as filter, true= spmv as kpi 

if SPMV_AS_KPI:
    print('\n\nEstrazione e ordinamento dei record con distanza media <= ' + str(DIST_TEMP_SOGLIA) + '\n')
    best_records = sorted(
        (tupla for tupla in records if tupla[DIST] <= DIST_TEMP_SOGLIA),
        key=lambda x: x[DIST]
    )
else:    
    print('\n\nEstrazione e ordinamento dei record con distanza media <= ' + str(DIST_TEMP_SOGLIA) + ' e |PMV| medio <= ' + str(DIST_SPMV_SOGLIA) + '\n')
    best_records = sorted(
        (tupla for tupla in records if tupla[DIST] <= DIST_TEMP_SOGLIA and tupla[KPI_PMV] <= DIST_SPMV_SOGLIA),
        key=lambda x: x[DIST]
    )
    
RECORDS_IF_EMPTY_SET=10
if not best_records:
    print('I filtri producono un dataset vuoto: Estrazione dei primi ' + str(RECORDS_IF_EMPTY_SET) + ' record\n')
    best_records = sorted((tupla for tupla in records), key=lambda x: x[DIST])
    best_records = best_records[:RECORDS_IF_EMPTY_SET]    
    
print_dataset(best_records, K)
    
best_records = getParetoSet(best_records, SPMV_AS_KPI)


best_records = sorted(
    (tupla for tupla in best_records),
    key=lambda x: -x[DIFF_ENERGY])

res=print_ParetoSet(best_records, records, lista, K)  
  

first_key, first_value = next(iter(res.items()))

if room == "020":
    output=first_value[0]
elif room == "001":
    output=first_value[0]

print(output)

