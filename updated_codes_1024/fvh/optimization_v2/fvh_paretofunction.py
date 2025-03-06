# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:34:36 2024

@author: utente
"""

import numpy as np
import pandas as pd
# from sPMV_v1 import *
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


###################################################    
#   TRACCIATO RECORD DATI INPUT
###################################################    
CSV_ID = 0                        # ID originale
CSV_ENERGY_CONS = 1               # consumo di energia  (KPI)
CSV_TIMESTAMP = 2                 # time stamp
CSV_INDOOR_T = 3                  # temperatura interna (campo di output)
CSV_INDOOR_RH = 4                 # umidità interna     (campo di outuput)
CSV_CO2 = 5                       # emissione (KPI)
CSV_OUTDOOR_T = 6                 # temperatura esterna (campo di input)
CSV_TSV = 7                       # comfort soggettivo (KPI)
CSV_PMV = 8                       # comfort oggettivo (KPI)


###################################################    
#   TRACCIATO RECORD LISTA DI LAVORO
###################################################    
PARETO_OPT = 0                # soluzione pareto ottimale
PROG = 1                      # progressivo di riga
DIST = 2                      # distanza del profilo di temperatura
KPI_ENERGY = 3                # energia nel periodo
KPI_CO2 = 4                   # emissione nel periodo
KPI_TSV = 5                   # comfort soggettivo nel periodo
KPI_PMV = 6                   # comfort oggettivo nel periodo
DIFF_ENERGY = 7               # scostamento percentuale tra energia prevista e energia consumata 
PERC_TSV = 8                  # percentuale dei periodi in confort
ENERGY_VALUES = 9             #tupla dei K consumi nel periodo
TEMP_VALUES = 10              #tupla delle K temperature nel periodo
PMV_VALUES = 11               #tupla dei K valori di PMV

SOGLIA_COMFORT = 0.5  


def split_day_night(df, h_start_work, h_stop_work ): #INPUT_CSV_FILE, 
    # df=df_or
    h_start_work=9
    h_stop_work=18
    # df.drop(columns=['room', 'floor'], axis=1,inplace=True)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df=df.round(2)

    ore=df['DATE'].dt.hour
    df['hour']=ore
    
    #split historical dataset into day_dataset (9:00 -> 18:00) and night_dataset (19:00 -> 8:00)
    df_work=[]
    df_night=[]

    for j in range(len(df)):

        if df['hour'][j]>=h_start_work and df['hour'][j]<=h_stop_work:
            df_work.append(df.values[j])
        else:
               df_night.append(df.values[j])

    df_work = pd.DataFrame(df_work, columns = ['ID', 'DATE', 'indoor T', 'indoor RH', 'co2', 'outT', 'value', 'tsv','spmv', 'hour'])
    new_column_order =['ID','value','DATE','indoor T', 'indoor RH','co2','outT', 'tsv', 'spmv', 'hour' ]
    df_work = df_work[new_column_order]
    df_night = pd.DataFrame(df_night, columns =  ['DATE', 'ID', 'value', 'indoor T', 'indoor RH', 'co2', 'outT','tsv', 'spmv', 'hour'])
    df_night = df_night[new_column_order]
    
    return df_work, df_night   

###################################################    
#   STAMPA SU FILE DELLA LISTA di TUPLE row
###################################################    
def print_row(file_path, row, mode):
   # Apri il file in modalità scrittura
    with open(file_path, mode, encoding='utf-8') as file:
        for elemento in row:
            file.write("".join(f"{elemento:<30}"))              
        
        file.write("\n")              

def leggi_e_memorizza_csv(df, printON):
    lista_tuple = []  # Lista per memorizzare le tuple
    
    # with open(file_path, mode='r', newline='', encoding='utf-8') as file:
    reader = df.to_dict(orient='records')
    
    # header = reader.fieldnames
    rows = [row for row in reader]

    # Stampa l'intestazione
    # if (printON):  
    #     print_row('input.txt', header, 'w')
    
    # Stampa le righe di dati e memorizza le tuple
    for row in rows:
        tupla = tuple(row.values())  # Converti la riga in tupla
        lista_tuple.append(tupla)    # Aggiungi la tupla alla lista
        if (printON):
            print_row('input.txt', tupla, 'a')       # Stampa la riga in formato tabellare    

    return lista_tuple

      


###################################################    
#   CREAZIONE DELLA LISTA DI K-TUPLE 
###################################################    

def unisci_tuple(lista, K):
    nuova_lista = []
    
    # Itera dalla i-esima tupla alla (len(lista) - K) per evitare errori di indice
    for i in range(len(lista) - K + 1):
        
        sum_energy = sum(float(lista[j][CSV_ENERGY_CONS]) for j in range(i, i + K))/K 
        sum_CO2 = sum(float(lista[j][CSV_CO2]) for j in range(i, i + K))/K 
        sum_TSV = sum(abs(float(lista[j][CSV_TSV])) for j in range(i, i + K))/K 
        sum_PMV = sum(abs(float(lista[j][CSV_PMV])) for j in range(i, i + K))/K 
        
        energy_values = [float(lista[j][CSV_ENERGY_CONS]) for j in range(i, i + K)]
        temp_values =   [float(lista[j][CSV_OUTDOOR_T]) for j in range(i, i + K)]     
        pmv_values =   [float(lista[j][CSV_PMV]) for j in range(i, i + K)]     

        
        # Unisci le tuple dalla i-esima alla (i + K)-esima
        tupla_unita = ('-', i, 0, sum_energy, sum_CO2, sum_TSV, sum_PMV, 0, 0, energy_values, temp_values, pmv_values)
        nuova_lista.append(tupla_unita)
        
    return nuova_lista


###################################################    
#   DISTANZA EUCLIDEA
###################################################    
def distanza_euclidea(vettore1, vettore2):
    #print(vettore1)
    #print(vettore2)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vettore1, vettore2)))


###################################################    
#   CALCOLO DELLA DISTANZA EUCLIDEA
###################################################    
def compute_temperature_dist(records, ENERGY_CONS, NEXT_TEMP, ak=1):

    lista_aggiornata = []
    energy_input = sum(v for v in ENERGY_CONS) 

    for tupla in records:
        # Estrai il sottovettore delle temperature
        sottovettore = tupla[TEMP_VALUES]

        # Calcola la distanza euclidea rispetto a `NEXT_TEMP`
        distanza = distanza_euclidea(sottovettore, NEXT_TEMP)/ak

        energy_tupla = sum(v for v in tupla[ENERGY_VALUES]) 
        diff_energy = (energy_input - energy_tupla)/energy_input
        
        perc_TSV = (sum(1 for x in tupla[PMV_VALUES] if abs(x) <= SOGLIA_COMFORT)) / ak
        
        # Crea una nuova tupla con la distanza nel secondo campo
        nuova_tupla = (tupla[PARETO_OPT], tupla[PROG], distanza, 
                       tupla[KPI_ENERGY], tupla[KPI_CO2], tupla[KPI_TSV], tupla[KPI_PMV], diff_energy , perc_TSV, 
                       tupla[ENERGY_VALUES], tupla[TEMP_VALUES], tupla[PMV_VALUES])

        #print(nuova_tupla) 
        
        # Aggiungi la nuova tupla alla lista aggiornata
        lista_aggiornata.append(nuova_tupla)
        
    return lista_aggiornata



###################################################    
#   CALCOLO DEL PARETO SET
###################################################    
def getParetoSet(records, spmv_key=False):

    lista = [list(t) for t in records]

    for t1 in lista:
        t1[PARETO_OPT] = '*'

    # Itera su tutte le tuple e controlla se sono dominate
    for i, t1 in enumerate(lista):
        dominata = False
        for j, t2 in enumerate(lista):
            #print(f"\nCoppia: {t1[PROG]} e {t2[PROG]}")
            if i != j and t1[DIST] >= t2[DIST] and t1[KPI_ENERGY] >= t2[KPI_ENERGY] and t1[KPI_CO2] >= t2[KPI_CO2] and t1[KPI_TSV] >= t2[KPI_TSV]: 
                if spmv_key:
                    if t1[KPI_PMV] >= t2[KPI_PMV]:
                        dominata = True
                        break
                else:
                    dominata = True
                    break
                        
                        
        if dominata:
            t1[PARETO_OPT] = ' '

            
    records = [tuple(t) for t in lista]
    return records



###################################################    
###################################################    
# def chart(records):

#     # Estrai i dati dai campi desiderati
#     x = [tupla[KPI_ENERGY] for tupla in records]  # Secondo campo
#     y = [tupla[KPI_CO2] for tupla in records]  # Terzo campo
#     z = [tupla[KPI_TSV] for tupla in records]  # Quarto campo

#     # Crea il grafico 3D
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Aggiungi i punti al grafico
#     ax.scatter(x, y, z)

#     # Imposta le etichette degli assi
#     ax.set_xlabel('Energy')
#     ax.set_ylabel('Emission')
#     ax.set_zlabel('TSV')

#     # Mostra il grafico
#     plt.show()


    
    
###################################################    
###################################################    
def print_dataset(data, ak, with_details=False, only_ParetoOptimal=False):

    
    if only_ParetoOptimal:
        tuples_to_plot = [t for t in data if t[PARETO_OPT] == "*"]
    else:
        tuples_to_plot = [t for t in data]
    

    print('\nValori medi relativi a', ak, 'periodi consecutivi\n')
    if with_details:
        print('frontier  ID        dist      KPI_E     KPI_CO2   KPI_TSV   KPI_PMV   SAVED_E   %TSV       (TEMP, CONSUMI, PMV)...')
        for tupla in tuples_to_plot:
            print("".join(f"{val:<10.2f}" if isinstance(val, float) else f"{val:<10}" for val in tupla[:ENERGY_VALUES]), end=" ")
            print("".join(f"({tupla[TEMP_VALUES][j]:<5.2f}, {tupla[ENERGY_VALUES][j]:<5.2f}, {tupla[PMV_VALUES][j]:<3.2f})" for j in range(ak)))
    else:
        print('frontier  ID        dist      KPI_E     KPI_CO2   KPI_TSV   KPI_PMV   SAVED_E   %TSV')
        for tupla in tuples_to_plot:
            print("".join(f"{val:<10.2f}" if isinstance(val, float) else f"{val:<10}" for val in tupla[:ENERGY_VALUES]))

            
            
            
###################################################    
###################################################    
# def plotParetoSet(best_records, records, lista, K):
   
#     tuples_to_plot = [t for t in best_records if t[PARETO_OPT] == "*"]

#     # Loop su ogni tupla da analizzare
#     for idx, t in enumerate(tuples_to_plot):


#         # Creazione della figura per il grafico a barre
#         fig, ax = plt.subplots(figsize=(10,3))
#         x = np.arange(K)  # Array per la posizione delle barre

    
#         # Larghezza delle barre
#         width = 0.15  

#         # Grafico delle barre
#         bars1 = ax.bar(x - width/2, NEXT_TEMP, width, label='PREVISIONE')
#         bars2 = ax.bar(x + width/2, t[FIRST_TEMP:FIRST_TEMP + K], width, label='DATO STORICO')

        
#         ax.set_ylim(min(min(t[FIRST_TEMP + i] for i in range(K)), min(NEXT_TEMP[i] for i in range(K))) - 3, max(max(t[FIRST_TEMP + i] for i in range(K)), max(NEXT_TEMP[i] for i in range(K))) + 3)
        
        
#         #csv_indoor_t_values = [records[t[PROG] + i][CSV_INDOOR_T] for i in range(K)]
#         #sv_indoor_rh_values = [records[t[PROG] + i][CSV_INDOOR_RH] for i in range(K)]
        
#         # Aggiunta di etichette
#         ax.set_title(f'tupla ID {lista[t[PROG]][CSV_ID]} dist = {t[DIST]:.2f}')
#         ax.set_xticks(x)
#         #ax.set_xticklabels([f'({NEXT_TEMP[i]}, {best_records[t][FIRST_TEMP + i]})' for i in range(K)])
#         ax.set_xticklabels([f'({NEXT_TEMP[i]:.2f}, {t[FIRST_TEMP + i]:.2f})' for i in range(K)])
#         ax.legend()
#         # Visualizzazione del grafico
#         plt.show() 
    
#         print('\nIl profilo suggerito di (temp, umidità) interna è:')
#         print(" ".join(f"({lista[int(t[PROG]) + i][CSV_INDOOR_T]}, {lista[int(t[PROG]) + i][CSV_INDOOR_RH]})" for i in range(K)))
#         print('\n========================================================\n\n\n')
#         # print(f"Average Temperature: {media_temp} °C\n")
#         # print(f"Average Humidity: {media_rh} %\n")

         
###################################################    
###################################################    
def print_ParetoSet(best_records, records, lista, K):
   
    tuples_to_plot = [t for t in best_records if t[PARETO_OPT] == "*"]
    result={}
    # message=""
    messages=[]
    # # Loop su ogni tupla da analizzare
    # for idx, t in enumerate(tuples_to_plot):
    #     print(f'Opzione {(idx + 1):3} (d {(t[DIST]):3.1f}):{(t[DIFF_ENERGY]*100):5.1f}% di energia risparmiata e comfort garantito per {(t[PERC_TSV]*100):5.1f}% del tempo: profilo [T,RH] nel periodo:' , end=" ")
    #     print(" ".join(f'[{float(lista[int(t[PROG]) + i][CSV_INDOOR_T]):5.1f}°C,{float(lista[int(t[PROG]) + i][CSV_INDOOR_RH]):5.1f}%]' for i in range(K)))
    # Loop su ogni tupla da analizzare
    for idx, t in enumerate(tuples_to_plot):
        
        media_temp = round((sum(lista[int(t[PROG]) + i][CSV_INDOOR_T] for i in range(K)) / K),1)
        media_rh = round((sum(lista[int(t[PROG]) + i][CSV_INDOOR_RH] for i in range(K)) / K),2)
        # print('Below different options for you!\n')
        # print(f'Option {(idx + 1):3}: \n 1) % Energy saved; {(t[DIFF_ENERGY]*100):5.1f}% \n 2) % of time where you are in comfort: {(t[PERC_TSV]*100):5.1f}% \n' , end=" ")
        print(f'RECOMMENDATION: The suggested temperature and humidity are:')
        print(f" Average Indoor Temperature: {media_temp} °C")
        print(f" Average Indoor Humidity: {media_rh} %\n")
        # print("".join(f'[{float(lista[int(t[PROG]) + i][CSV_INDOOR_T]):5.1f}°C,{float(lista[int(t[PROG]) + i][CSV_INDOOR_RH]):5.1f}%]' for i in range(K)))
        print(' Please, adjust the indoor conditions accordingly.')
        if t[DIFF_ENERGY]*100<0:
            message = (
                f'1) % Energy saved: --- \n2) % of time where you are in comfort: {(t[PERC_TSV]*100):5.1f}% \n' #Option {(idx + 1):3}: \n 
                f'RECOMMENDATION: The suggested temperature and relative humidity are:'
                f"Indoor Temperature: {media_temp} °C"
                f"\nIndoor Humidity: {media_rh} %\n"
                # print("".join(f'[{float(lista[int(t[PROG]) + i][CSV_INDOOR_T]):5.1f}°C,{float(lista[int(t[PROG]) + i][CSV_INDOOR_RH]):5.1f}%]' for i in range(K)))
                '\nPlease, adjust the indoor conditions accordingly.'
            )
        else:
        
            message = (
               f'1) % Energy saved: {(t[DIFF_ENERGY]*100):5.1f}% \n2) % of time where you are in comfort: {(t[PERC_TSV]*100):5.1f}% \n' #Option {(idx + 1):3}: \n 
               f'RECOMMENDATION: The suggested temperature and relative humidity are:'
               f"Indoor Temperature: {media_temp} °C"
               f"\nIndoor Humidity: {media_rh} %\n"
               # print("".join(f'[{float(lista[int(t[PROG]) + i][CSV_INDOOR_T]):5.1f}°C,{float(lista[int(t[PROG]) + i][CSV_INDOOR_RH]):5.1f}%]' for i in range(K)))
               '\nPlease, adjust the indoor conditions accordingly.'
            )
        
        messages.append(message.strip())
        
    # Convert the single message into a dictionary
    result_dict = {
        "messages": messages  # Lista di messaggi separati
    }
        
    return result_dict
