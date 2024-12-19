# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:23:35 2024

@author: utente
"""
from fvh_paretofunction import *
from API_forecast_fvh_byntua import *

INPUT_CSV_FILE = 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/s3.3.1_codes_v2/updated_codes_1024/fvh/optimization_v1/optimization_v2/dataset/processed_floor1/processed_fvh_ntua_room129_91224.csv'  # dataset 
df_or=pd.read_csv(INPUT_CSV_FILE, sep=',')
df_or.drop(columns=['room', 'floor'], axis=1,inplace=True)
df_or['DATE']=pd.to_datetime(df_or['DATE'])
df_or=df_or.drop(columns=['index'])

#controlla se ci sono colonne unnamed e in caso droppale
unnamed_columns = [col for col in df_or.columns if col.startswith('Unnamed')]
if unnamed_columns:
    print(f"Dropping columns: {unnamed_columns}")
    df_or.drop(columns=unnamed_columns, inplace=True)
else:
    print("No 'Unnamed' columns found. Proceeding without changes.")


columns_with_nan = df_or.columns[df_or.isnull().any()].tolist()
print(f"Colonne con NaN: {columns_with_nan}")
rows_with_nan = df_or[df_or.isnull().any(axis=1)]
print("Righe con NaN:")
print(rows_with_nan)


df_work=[]
df_night=[]
h_start_work=9
h_stop_work=18

df_work, df_night= split_day_night(df_or, h_start_work, h_stop_work)


#once integrated the API for weather forecast and energy forecast, these lines can be deleted
df_to_use=df_work[0:len(df_work)-10].reset_index(drop=True)
df_next_day=df_work[len(df_work)-10: len(df_work)].reset_index(drop=True)


K =10        
NEXT_TEMP = tuple(round(df_next_day['outT'],2))   

#request for energy forecast data (API by NTUA)
token = gen_token(username, password)
#the section and floor should be changed based on the considered room
response=get_data(token,section='C', floor='1') 
first_10_values = response["Floors"]["C_1"][:10] 
# first_10_values = response["Rooms"]["JKC2.1"][:10] #per consumi su sezione C piano 2 room 1

ENERGY_CONS = tuple(np.round([list(d.values())[0] for d in first_10_values],2))
                                   
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
best_records = sorted(
    (tupla for tupla in best_records),
    key=lambda x: -x[DIFF_ENERGY])

print_ParetoSet(best_records, records, lista, K)    

res=print_ParetoSet(best_records, records, lista, K)    
first_key, first_value = next(iter(res.items()))
output=first_value[0]
print(output)

