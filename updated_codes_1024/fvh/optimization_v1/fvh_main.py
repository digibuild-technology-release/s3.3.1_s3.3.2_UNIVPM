# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:23:35 2024

@author: utente
"""
from fvh_paretofunction import *

#upload the csv file
INPUT_CSV_FILE = 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/fvh/optimiz/dataset/processed_fvh_ntua_room117_91224.csv'  # dataset 
#select the room between 117, 121, 333, 326
room='117'
df_or=pd.read_csv(INPUT_CSV_FILE, sep=',')
df_or.drop(columns=['room', 'floor'], axis=1,inplace=True)
df_or['DATE']=pd.to_datetime(df_or['DATE'])

#checks: Nan and negative values
unnamed_columns = [col for col in df_or.columns if col.startswith('Unnamed')]
if unnamed_columns:
    print(f"Dropping columns: {unnamed_columns}")
    df_or.drop(columns=unnamed_columns, inplace=True)
else:
    print("No 'Unnamed' columns found. Proceeding without changes.")


if (df_or['value'] < 0).any():
    print("Ci sono valori negativi nella colonna 'value'")
else:
    print("Non ci sono valori negativi nella colonna 'value'")

df=df_or
df=df.reset_index(drop=True)

df_work=[]
df_night=[]
h_start_work=9
h_stop_work=18

df_work, df_night= split_day_night(df, h_start_work, h_stop_work)


df_to_use=df_work[0:len(df_work)-10].reset_index(drop=True)
df_next_day=df_work[len(df_work)-10: len(df_work)].reset_index(drop=True)

K =10        
#to be modified with weather forecast data from datasharing             
# NEXT_TEMP = (5.05, 5.32, 5.92, 6.37, 6.65, 6.72, 6.6, 6.45, 6.38, 6.5)
NEXT_TEMP=tuple(round(df_next_day['outT'],2)) 
#to be modified with energy forecasting API              
ENERGY_CONS = tuple(round(df_next_day['value'],2))                                      
# ENERGY_CONS =(0.46, 0.46, 0.46, 0.46, 0.46, 0.22, 0.22, 0.22, 0.22)               
                                           
DIST_TEMP_SOGLIA = 0.7                                              
DIST_SPMV_SOGLIA = 0.7                                             

lista = leggi_e_memorizza_csv(df_to_use, False)

# creazione delle tuple di lunghezza K
records = unisci_tuple(lista, K)


records = compute_temperature_dist(records, ENERGY_CONS,NEXT_TEMP, K)

print_dataset(records, K, True)


SPMV_AS_KPI = False #false = spmv come filtro, true= spm come kpi 

if SPMV_AS_KPI:
    best_records = sorted(
        (tupla for tupla in records if tupla[DIST] <= DIST_TEMP_SOGLIA),
        key=lambda x: x[DIST]
    )
else:    
    best_records = sorted(
        (tupla for tupla in records if tupla[DIST] <= DIST_TEMP_SOGLIA and tupla[KPI_PMV] <= DIST_SPMV_SOGLIA),
        key=lambda x: x[DIST]
    )
    
RECORDS_IF_EMPTY_SET=10
if not best_records:
    best_records = sorted((tupla for tupla in records), key=lambda x: x[DIST])
    best_records = best_records[:RECORDS_IF_EMPTY_SET]    
    
print_dataset(best_records, K)
    
best_records = getParetoSet(best_records, SPMV_AS_KPI)

    
print_ParetoSet(best_records, records, lista, K)    


best_records = sorted(
    (tupla for tupla in best_records),
    key=lambda x: -x[DIFF_ENERGY])

print_ParetoSet(best_records, records, lista, K)    


best_records = sorted(
    (tupla for tupla in best_records),
    key=lambda x: -x[PERC_TSV])

res=print_ParetoSet(best_records, records, lista, K)    
first_key, first_value = next(iter(res.items()))

if room == "333":
    output=first_value[23]
elif room == "326":
    output=first_value[66]
elif room == "117":
    output=first_value[36]
elif room == "121":
    output=first_value[23]

print(output)