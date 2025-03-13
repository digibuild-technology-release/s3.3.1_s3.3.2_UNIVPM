import requests, json, logging, sys
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional

# Importiamo le funzioni di data_acquisition
from dataroom import process_in_batches, gen_token, load_sensors_from_excel, get_unique_rooms, get_sensor_ids

# Funzioni di utilità
from processing_functions import dataset_input, denorm
from sPMV_v1 import sPMV_calculation

# Inizializza FastAPI
app = FastAPI(title="Indoor Comfort Predictions API", version="1.0.0")

# Configura il logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Percorsi modelli
model_tin_path = os.path.join(os.path.dirname(__file__), 'models', 'tin_pred_fvh_191124.h5')
model_rh_path = os.path.join(os.path.dirname(__file__), 'models', 'rh_pred_fvh_201124.h5')
model_tsv_path = os.path.join(os.path.dirname(__file__), 'models', 'RF_tsv.pkl')

def clean_dataframe(df):
    # Rimuove o sostituisce i valori infiniti
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Sostituisce NaN con la media della colonna (o altre logiche)
    df.fillna(df.mean(), inplace=True)
    return df

# Caricamento modelli
def load_models():
    try:
        model_tin = load_model(model_tin_path)
        model_rh = load_model(model_rh_path)
        tsv_model = pickle.load(open(model_tsv_path, "rb"))
        return model_tin, model_rh, tsv_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

model_tin, model_rh, tsv_model = load_models()

# Caricamento variabili d'ambiente
load_dotenv()
username = os.getenv("KEYCLOACK_USERNAME")
password = os.getenv("KEYCLOACK_PASSWORD")

df_sensors = load_sensors_from_excel("sensor_room.xlsx")

# Ottieni i dati dei sensori per ogni stanza
def get_room_dataframes():
    room_dataframes = {}
    unique_rooms = get_unique_rooms(df_sensors)
    
    for room in unique_rooms:
        sensors = get_sensor_ids(df_sensors, room)
        if sensors:
            df_room = process_in_batches({
                'sensors': sensors,
                'pilot_db': 'fvh',
                'range': {'value': 7, 'subject': 'days'}
            }, token, df_sensors)
            if not df_room.empty:
                room_dataframes[room] = df_room
    return room_dataframes

@app.get("/predict_dynamic")
async def predict_dynamic():
    try:
        token = gen_token(username, password)
        room_dataframes = get_room_dataframes()
        if not room_dataframes:
            raise HTTPException(status_code=500, detail="No sensor data available for the rooms.")
        
        df_merged = pd.concat(room_dataframes.values(), ignore_index=True)
        df_merged["datetime"] = pd.to_datetime(df_merged["datetime"])
        df_merged.set_index("datetime", inplace=True)
        
        df_numeric = df_merged.select_dtypes(include=[np.number])
        # Pulizia dei dati per rimuovere valori infiniti e NaN prima della predizione
        df_numeric = clean_dataframe(df_numeric)
        df_resampled = df_numeric.resample("h").mean().interpolate()
        df_resampled = df_merged.join(df_resampled, rsuffix='_resampled').reset_index()
        
        indoor_T_cols = [col for col in df_resampled.columns if col.endswith("_IndoorTemperature")]
        indoor_RH_cols = [col for col in df_resampled.columns if col.endswith("_IndoorRH")]
        outT_cols = [col for col in df_resampled.columns if col.endswith("_OutdoorTemperature")]
        electricity_cols = [col for col in df_resampled.columns if col.endswith("_electricity")]
        CO2_cols = [col for col in df_resampled.columns if col.endswith("_Co2")]
        
        df_resampled["indoor_T_avg"] = df_resampled[indoor_T_cols].mean(axis=1, skipna=True) if indoor_T_cols else np.nan
        df_resampled["indoor_RH_avg"] = df_resampled[indoor_RH_cols].mean(axis=1, skipna=True) if indoor_RH_cols else np.nan
        df_resampled["outT_avg"] = df_resampled[outT_cols].mean(axis=1, skipna=True) if outT_cols else np.nan
        df_resampled["electricity_avg"] = df_resampled[electricity_cols].mean(axis=1, skipna=True) if electricity_cols else np.nan
        df_resampled["CO2_avg"] = df_resampled[CO2_cols].mean(axis=1, skipna=True) if CO2_cols else np.nan
        
        # Pulizia dei valori NaN
        df_resampled = clean_dataframe(df_resampled)
        
        # Riempie i valori NaN con la media della colonna
        df_resampled[indoor_T_cols] = df_resampled[indoor_T_cols].fillna(df_resampled[indoor_T_cols].mean())
        df_resampled[indoor_RH_cols] = df_resampled[indoor_RH_cols].fillna(df_resampled[indoor_RH_cols].mean())
        df_resampled[outT_cols] = df_resampled[outT_cols].fillna(df_resampled[outT_cols].mean())
        df_resampled[electricity_cols] = df_resampled[electricity_cols].fillna(df_resampled[electricity_cols].mean())
        df_resampled[CO2_cols] = df_resampled[CO2_cols].fillna(df_resampled[CO2_cols].mean())
          
        df_resampled.rename(columns={'datetime': 'date'}, inplace=True)
        #print("df_resampled", df_resampled)
        input_datasetor = dataset_input(df_resampled)
        input_dataset = input_datasetor[2:2+168].reset_index(drop=True)
        
        selected_cols = ["indoor_T_avg", "outT_avg", "outNext", "day_of_week_sin", "day_of_week_cos", "hour_sin", "hour_cos"]
        Xt = input_dataset[selected_cols]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(Xt)
        X_seqt = scaled_data.reshape(1, 168, 7)
        y_predT = model_tin.predict(X_seqt)
        yPredDt = np.round(denorm(y_predT, Xt, scaler).reshape(24,), 1)
        
        Xrh = input_dataset[["indoor_RH_avg", "outT_avg", "outNext", "day_of_week_sin", "day_of_week_cos", "hour_sin", "hour_cos"]]
        scaled_datarh = scaler.fit_transform(Xrh)
        X_seqrh = scaled_datarh.reshape(1, 168, 7)
        y_predRH = model_rh.predict(X_seqrh)
        yPredDrh = np.round(denorm(y_predRH, Xrh, scaler).reshape(24,), 1)
        
        timestamps = pd.date_range(start=input_dataset["date"].iloc[0], periods=24, freq="H")
        inputcomfortcalc = pd.DataFrame({
            "date": timestamps,
            "pred_indoorT": yPredDt.reshape(24,),
            "pred_indoorRH": yPredDrh.reshape(24,)
        })
        spmv_pred = sPMV_calculation(inputcomfortcalc["pred_indoorT"], inputcomfortcalc["pred_indoorRH"], inputcomfortcalc["date"])
        mean_spmv24 = round(np.mean(spmv_pred["sPMV"]), 1)
        
        X_now = pd.DataFrame({
          "indoor T_y": input_dataset["indoor_T_avg"],
          "indoor RH_y": input_dataset["indoor_RH_avg"],
          "outT": input_dataset["outT_avg"]
        })
        #print("X_now", X_now)
        tsv_now = np.round(tsv_model.predict(X_now), 2)
        mean_tsv_now = round(np.mean(tsv_now), 1)
        
         # Predizione TSV +24h
        t = timestamps[0]
        ind = input_datasetor[input_datasetor["date"] == t]
        i = ind.index
        perout = input_datasetor["outNext"].iloc[i.values[0]: (i.values[0]) + 24]
        X24 = pd.DataFrame({
            "indoor T_y": yPredDt.reshape(24,),
            "indoor RH_y": yPredDrh.reshape(24,),
            "outT": perout
        })
        
        #print("X_24", X24)
        #print("tsv_model.predict(X24)", tsv_model.predict(X24))
        mean_tsv_tomorrow = round(np.mean(tsv_model.predict(X24)), 1)
        #print("mean_tsv_tomorrow ", mean_tsv_tomorrow )

        # Calcoliamo la media delle previsioni
        mean_temp = np.mean(yPredDt).round(1)
        mean_rh = np.mean(yPredDrh).round(1)

        # Risultati finali
        return {
            "Indoor Temperature Prediction (Next 24h)": mean_temp,
            "Indoor RH Prediction (Next 24h)": mean_rh,
            "Mean sPMV (Next 24h)": mean_spmv24,
            "Mean TSV (Current)": mean_tsv_now,
            "Mean TSV (Next 24h)": mean_tsv_tomorrow
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
