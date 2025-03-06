
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:37:43 2024

@author: utente
"""
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

CSV_ID = 0                        # ID originale
CSV_ENERGY_CONS = 1               # consumo di energia  (KPI)
CSV_TIMESTAMP = 2                 # time stamp
CSV_INDOOR_T = 3                  # temperatura interna (campo di output)
CSV_OUTDOOR_T = 4                 # temperatura esterna (campo di input)
CSV_TSV = 5                       # comfort (KPI)

PARETO_OPT = 0                
PROG = 1                     
DIST = 2                      
KPI_ENERGY = 3                
KPI_TSV = 4                  
FIRST_TEMP = 5
###################################################    
#   STAMPA SU FILE DELLA LISTA di TUPLE row
###################################################    
def print_row(file_path, row, mode):
   # Apri il file in modalità scrittura
    with open(file_path, mode, encoding='utf-8') as file:
        for elemento in row:
            file.write("".join(f"{elemento:<30}"))              
        file.write("\n")              


     
def split_day_night(df, h_start_work, h_stop_work ): #INPUT_CSV_FILE, 
    # INPUT_CSV_FILE ='C:/Users/utente/OneDrive - Università Politecnica delle Marche/Desktop/OTTIMIZZAZIONE_COLLAB/per marinelli/datasets/processed_fvh_ntua_TSVml_room117.csv'
    # df=pd.read_csv(INPUT_CSV_FILE, sep=',')

    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M")

    df=df[['ID','value', 'date', 'indoor T','outdoorT', 'TSV' ]]
    df=df.round(2)

    ore=df['date'].dt.hour
    df['hour']=ore
    
    # df.set_index('DATE', inplace=True)  # Set the timestamp column as index
    # # Resample to hourly, calculating the mean value for each hour
    # df = df.resample('H').mean().interpolate()
    # df=df.reset_index()
    
    #split historical dataset into day_dataset (9:00 -> 18:00) and night_dataset (19:00 -> 8:00)
    df_work=[]
    df_night=[]

    for j in range(len(df)):

        if df['hour'][j]>=h_start_work and df['hour'][j]<=h_stop_work:
            df_work.append(df.values[j])
        else:
               df_night.append(df.values[j])

    df_work = pd.DataFrame(df_work, columns = ['ID', 'value', 'date', 'indoor T ', 'outT','tsv', 'hour'])
    df_night = pd.DataFrame(df_night, columns = ['ID', 'value', 'date', 'indoor T ', 'outT',
           'tsv', 'hour'])
    return df_work, df_night


###################################################    
#   LETTURA DEI DATI DA FILE CSV
###################################################  

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
        
        sum_energy = sum(float(lista[j][CSV_ENERGY_CONS]) for j in range(i, i + K)) 
        # sum_CO2 = sum(float(lista[j][CSV_CO2]) for j in range(i, i + K)) 
        sum_TSV = sum(abs(float(lista[j][CSV_TSV])) for j in range(i, i + K)) 
        
        # Unisci le tuple dalla i-esima alla (i + K)-esima
        tupla_unita = ('-', i,) + (0, sum_energy,  sum_TSV) + tuple(float(lista[j][CSV_OUTDOOR_T]) for j in range(i, i + K)) #sum_CO2,
        nuova_lista.append(tupla_unita)
        
    return nuova_lista


###################################################    
#   DISTANZA EUCLIDEA
###################################################    
def distanza_euclidea(vettore1, vettore2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vettore1, vettore2)))


###################################################    
#   CALCOLO DELLA DISTANZA EUCLIDEA
###################################################    
def compute_temperature_dist(records, NEXT_TEMP):

    lista_aggiornata = []
    for tupla in records:
        # Estrai il sottovettore delle temperature
        sottovettore = tupla[FIRST_TEMP:]

        # Calcola la distanza euclidea rispetto a `NEXT_TEMP`
        distanza = distanza_euclidea(sottovettore, NEXT_TEMP)

        # Crea una nuova tupla con la distanza nel secondo campo
        nuova_tupla = (tupla[PARETO_OPT], tupla[PROG], distanza) + tupla[KPI_ENERGY:]

        # Aggiungi la nuova tupla alla lista aggiornata
        lista_aggiornata.append(nuova_tupla)

    return lista_aggiornata



###################################################    
#   CALCOLO DEL PARETO SET
###################################################    
def getParetoSet(records):

    # Converti la lista in una lista di liste per renderla mutabile
    lista = [list(t) for t in records]

    for t1 in lista:
        t1[PARETO_OPT] = '*'

        
    # Itera su tutte le tuple e controlla se sono dominate
    for i, t1 in enumerate(lista):
        dominata = False
        for j, t2 in enumerate(lista):
            #print(f"\nCoppia: {t1[PROG]} e {t2[PROG]}")
            if i != j and t1[DIST] >= t2[DIST] and t1[KPI_ENERGY] >= t2[KPI_ENERGY]  and t1[KPI_TSV] >= t2[KPI_TSV]:  # confronta il terzo campo (indice 2) and t1[KPI_CO2] >= t2[KPI_CO2]
                dominata = True
                break

        if dominata:
            t1[PARETO_OPT] = ' '

            
    # Converti di nuovo in tuple immutabili, se necessario
    records = [tuple(t) for t in lista]
    return records



###################################################    
###################################################    
def chart(records):

    # Estrai i dati dai campi desiderati
    x = [tupla[KPI_ENERGY] for tupla in records]  # Secondo campo
    # y = [tupla[KPI_CO2] for tupla in records]  # Terzo campo
    y = [tupla[KPI_TSV] for tupla in records]  # Quarto campo

    # Crea il grafico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Aggiungi i punti al grafico
    ax.scatter(x, y)

    # Imposta le etichette degli assi
    ax.set_xlabel('Energy')
    # ax.set_ylabel('Emission')
    ax.set_zlabel('TSV')

    # Mostra il grafico
    plt.show()


    
def plotParetoSetUCL(best_records, records, lista,  K): #NEXT_TEMP,
    tuples_to_plot = [t for t in best_records if t[PARETO_OPT] == "*"]
    messages=[]
    # Loop su ogni tupla da analizzare
    for idx, t in enumerate(tuples_to_plot):

        media_temp = round((sum(float(lista[int(t[PROG]) + i][CSV_INDOOR_T]) for i in range(K)) / K),2)
        # media_rh = round((sum(float(lista[int(t[PROG]) + i][CSV_INDOOR_RH]) for i in range(K)) / K),2)

        # print('\nThe profile of indoor temperature and indoor humidity suggested to the end-user is:\n')
        # # print(" ".join(f"({lista[int(t[PROG]) + i][CSV_INDOOR_T]}, {lista[int(t[PROG]) + i][CSV_INDOOR_RH]})" for i in range(K)))
        # # print('\nThe mean indoor temperature and relative humidity are:')
        # print(f"Average Temperature: {media_temp} °C\n")
        # print(f"Average Humidity: {media_rh} %\n")
        # print('\n Please, adjust the indoor conditions accordingly.')
        # print('\n========================================================\n\n\n')
        
        message = (
            f"The indoor temperature suggested to the end-user is:\n"
            f"Average Temperature: {media_temp} °C\n"
            # f"Average Humidity: {media_rh} %\n"
            "Please, adjust the indoor conditions accordingly.\n"
            
        )
        messages.append(message.strip())
    # Convert the single message into a dictionary
    result_dict = {
        
        "full_message": messages#.strip()  # Full concatenated message as a single entry
    }

    return result_dict






    
    
    
    
    
    # result={}
    # message=""
    # tuples_to_plot = [t for t in best_records if t[PARETO_OPT] == "*"]

    # for idx, t in enumerate(tuples_to_plot):

    #     fig, ax = plt.subplots(figsize=(10,3))
    #     x = np.arange(K) 

    #     width = 0.15  
    #     bars1 = ax.bar(x - width/2, NEXT_TEMP, width, label='prediction')
    #     bars2 = ax.bar(x + width/2, t[FIRST_TEMP:FIRST_TEMP + K], width, label='historical data')

    #     ax.set_ylim(min(min(t[FIRST_TEMP + i] for i in range(K)), min(NEXT_TEMP[i] for i in range(K))) - 3, max(max(t[FIRST_TEMP + i] for i in range(K)), max(NEXT_TEMP[i] for i in range(K))) + 3)
        
    
    #     ax.set_xticks(x)
    #     ax.set_xticklabels([f'({NEXT_TEMP[i]:.2f}, {t[FIRST_TEMP + i]:.2f})' for i in range(K)])
    #     ax.legend()
    #     plt.show() 
    #     media_temp = round((sum(lista[int(t[PROG]) + i][CSV_INDOOR_T] for i in range(K)) / K),2)
    #     # media_rh = round((sum(lista[int(t[PROG]) + i][CSV_INDOOR_RH] for i in range(K)) / K),2)
        
    #     message += (
    #         f"The indoor temperature suggested to the end-user is:\n"
    #         f"Average Temperature: {media_temp} °C\n"
    #         # f"Average Humidity: {media_rh} %\n"
    #         "Please, adjust the indoor conditions accordingly.\n"
            
    #     )

    #     # Convert the single message into a dictionary
    #     result_dict = {
            
    #         "full_message": message.strip()  # Full concatenated message as a single entry
    #     }
    