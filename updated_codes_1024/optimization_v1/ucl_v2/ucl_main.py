# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:23:35 2024

@author: utente
"""
from ucl_paretofunction import *
# from adaptive_function import adaptive_model
from api_client_auth import *


INPUT_CSV_FILE = 'C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/OTTIMIZZAZIONE_COLLAB/per marinelli/ucl/alldata_room001_tsv.csv' 

df_or=pd.read_csv(INPUT_CSV_FILE, sep=',')
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
df_to_use=df_work[0:len(df_work)-10]
df_next_day=df_work[len(df_work)-10: len(df_or)]


#weather forecast from BBI     
NEXT_TEMP = tuple(round(df_next_day['outT'],2))   
          
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
room = "001"  
valori = list(filter(lambda d: room in d, predictions)) #here 020 should be changed with the id of the current room
ENERGY_CONS = [d[room] for d in valori][h_start_work:h_stop_work] #here 020 should be changed with the id of the current room
ENERGY_CONS = tuple(map(lambda x: round(x, 2), ENERGY_CONS))
                                   
                                          
DIST_TEMP_SOGLIA = 0.7                                              
DIST_SPMV_SOGLIA = 0.7                                             

lista = leggi_e_memorizza_csv(df_work, False)

# creazione delle tuple di lunghezza K
records = unisci_tuple(lista, K)

#print_dataset(records, K, True)

# print('\n\nCalcolo degli scostamenti da:\n\ttemperature previste ' + str(NEXT_TEMP) + '\n\t  e consumi previsti ' + str(ENERGY_CONS) + '\n')
records = compute_temperature_dist(records, ENERGY_CONS,NEXT_TEMP, K)

print_dataset(records, K, True)


SPMV_AS_KPI = False #false = spmv come filtro, true= spm come kpi 

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

if room == "020":
    output=first_value[-2]
elif room == "001":
    output=first_value[-6]

print(output)

