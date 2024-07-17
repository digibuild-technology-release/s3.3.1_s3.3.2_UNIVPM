# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:03:21 2024

@author: utente
"""

import pandas as pd
from influxdb_client import InfluxDBClient
import matplotlib.pyplot as plt
import numpy  as np
import requests
# from sPMV_v1 import sPMV_calculation

# InfluxDB 2.0 Credentials
influx_url = "https://influx.fvh.io"
influx_org = "Energy"
influx_bucket = "Nuuka_7683"
influx_token = "S4T_4IPUUnugYf3ea15r1fne3NDwHeyCVav4EuUnSmFX4g49dmR54GUU0_xpnGV5j8Wv6XrPA-aLCEPTwewGng=="

# Create InfluxDB client. Use enable_gzip=True to enable gzip compression and reduce download times.
client = InfluxDBClient(url=influx_url, token=influx_token, enable_gzip=True)


def query_measurements():
    """Query InfluxDB for list of measurements"""
    measurement_query = f"""
    import "influxdata/influxdb/schema"
    schema.measurements(bucket: "{influx_bucket}", start: -1y)
    """
    print(f"Query:\n {measurement_query}")
    query_api = client.query_api()
    tables = query_api.query(query=measurement_query, org=influx_org)
    # Print just measurements
    measurements = [row.values["_value"] for table in tables for row in table]
    print("Measurements:\n{}".format("\n".join(measurements)))


def query_meta():
    """Query InfluxDB for list of all datapoints of building 7683"""
    meta_query = f"""
    from(bucket: "{influx_bucket}")
        |> range(start:-1y)
        |> filter(fn: (r) => r._measurement == "measurement_info_7683")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    query_api = client.query_api()
    result = query_api.query_data_frame(query=meta_query, org=influx_org)
    # For some reason the result is in separate tables, so concatenate all tables into one dataframe
    df = pd.concat([result], ignore_index=True)
    # Drop unnecessary columns like _start, _stop
    df = df.drop(columns=["_start", "_stop", "result", "table", "_time", "_measurement"])
    return df


def query_data(datapointids: list):
    # Query InfluxDB for list of some data
    # Create filter for datapointids
    filter = " or ".join([f'r.datapointid == "{datapointid}"' for datapointid in datapointids])
    data_query = f"""
    from(bucket: "{influx_bucket}")
        |> range(start:-1y)
        |> filter(fn: (r) => r._measurement == "nuuka_7683" )
        |> filter(fn: (r) => {filter})
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    # print(f"Query:\n {data_query}")
    query_api = client.query_api()
    df = query_api.query_data_frame(query=data_query, org=influx_org)
    # Drop unnecessary columns like _start, _stop
    df = df.drop(columns=["_start", "_stop", "result", "table", "_measurement"])
    # Make _time column the index and rename it to time
    df = df.set_index("_time").rename_axis("time")
    return df

# Function to get historical weather data
def get_historical_weather(api_key, lat, lon, start_date, end_date):
    url = f'https://api.weatherbit.io/v2.0/history/hourly?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&key={api_key}&include=minutely'
    response = requests.get(url)
    data = response.json()
    return data


def create_sequences(input_data, sequence_length, forecast_horizon):
    X_sequences, y_sequences = [], []
    for k in range(len(input_data) - sequence_length - forecast_horizon + 1):
        X_sequences.append(input_data[k:k + sequence_length])
    return np.array(X_sequences)