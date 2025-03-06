from fastapi import FastAPI, HTTPException
import os
import pandas as pd
import numpy as np
from fvh_paretofunction import *
from API_forecast_fvh_byntua import *
from API_weatherforecast_helsinki import *
from dotenv import load_dotenv
from datetime import datetime

# Caricamento variabili d'ambiente
load_dotenv()
username = os.getenv("KEYCLOACK_USERNAME")
password = os.getenv("KEYCLOACK_PASSWORD")

app = FastAPI()

# Configurazione della cartella principale
INPUT_DIR = 'dataset/'

@app.get("/optimize/{file_number}")
def process_file(file_number: int):
    print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), "start")

    # Determina piano e sezione in base al file_number (o logica personalizzata)
    # Per esempio, se file_number è 375, il piano è 2 e la sezione è 'C'
    floor, section = get_floor_and_section(file_number)
    
    target_file = f"{file_number}.csv"
    csv_files = []

    # Cerca solo il file specifico dentro tutte le sottocartelle
    for root, _, files in os.walk(INPUT_DIR):
        if target_file in files:
            csv_files.append(os.path.join(root, target_file))

    if not csv_files:
        raise HTTPException(status_code=404, detail=f"File {target_file} non trovato nella cartella {INPUT_DIR}")

    file_path = csv_files[0]
    print(f"Processing file: {file_path}")

    # Lettura del dataset
    df_or = pd.read_csv(file_path, sep=',')
    df_or.drop(columns=['room', 'floor', 'index'], errors='ignore', inplace=True)
    df_or['DATE'] = pd.to_datetime(df_or['DATE'], errors='coerce')

    unnamed_columns = [col for col in df_or.columns if col.startswith('Unnamed')]
    if unnamed_columns:
        df_or.drop(columns=unnamed_columns, inplace=True)

    columns_with_nan = df_or.columns[df_or.isnull().any()].tolist()
    if columns_with_nan:
        return {"error": f"Colonne con NaN: {columns_with_nan}", "data": df_or[df_or.isnull().any(axis=1)].to_dict()}

    h_start_work, h_stop_work = 9, 18
    df_work, df_night = split_day_night(df_or, h_start_work, h_stop_work)
    df_to_use = df_work

    K = 10
    a, b = get_weather_forecast_FVH()
    NEXT_TEMP = b['outT_next_1day'][9:19]

    token = gen_token(username, password)
    response = get_data(token, section=section, floor=floor)  # Usa i parametri dinamici
    print(response)
    first_10_values = response.get("Floors", {}).get(f"{section}_{floor}", [])[:10]
    ENERGY_CONS = tuple(np.round([list(d.values())[0] for d in first_10_values], 2)) if first_10_values else ()

    if len(ENERGY_CONS) < K or len(NEXT_TEMP) < K:
        raise ValueError("Dati insufficienti per la previsione (meno di 10 valori trovati).")

    DIST_TEMP_SOGLIA = 0.7
    DIST_SPMV_SOGLIA = 0.7

    lista = leggi_e_memorizza_csv(df_to_use, False)
    records = unisci_tuple(lista, K)
    records = compute_temperature_dist(records, ENERGY_CONS, NEXT_TEMP, K)

    SPMV_AS_KPI = False
    best_records = sorted((r for r in records if r[DIST] <= DIST_TEMP_SOGLIA and (SPMV_AS_KPI or r[KPI_PMV] <= DIST_SPMV_SOGLIA)),
                          key=lambda x: x[DIST])

    if not best_records:
        best_records = sorted(records, key=lambda x: x[DIST])[:10]

    best_records = getParetoSet(best_records, SPMV_AS_KPI)
    best_records = sorted(best_records, key=lambda x: -x[DIFF_ENERGY])

    res = print_ParetoSet(best_records, records, lista, K)
    first_key, first_value = next(iter(res.items()))
    output = first_value[0]
    
    print(output)

    print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), "end")

    return {
        "file": target_file,
        "output": output,
        "timestamp": datetime.now().isoformat()
    }

# Funzione per determinare piano e sezione
def get_floor_and_section(file_number: int):
    # Mappatura esempio (modifica in base alla logica desiderata)
    if file_number == 129:
        return 1, 'C'
    elif file_number == 375:
        return 2, 'C'
    elif file_number == 378:
        return 2, 'C'
    elif file_number == 379:
        return 2, 'C'
    elif file_number == 380:
        return 2, 'C'
    elif file_number == 326:
        return 4, 'B'
    elif file_number == 333:
        return 4, 'B'
    elif file_number == 121:
        return 5, 'B'
    else:
        return 6, 'B'
