# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:26:28 2024

@author: utente
"""

import requests, json, logging, sys, os
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def gen_token(username, password):
    '''
        Purpose: Generates an authentication token by making a POST request to a specific URL with the provided username and password.
    Parameters:
        username: The username for authentication.
        password: The password for authentication.
    Process:
        Constructs a payload with authentication details and headers.
        Makes a POST request to the authentication URL.
        Parses the JSON response to extract the access_token.
    Returns: The access token as a string.
    '''
    url = "https://digibuild.epu.ntua.gr/auth/realms/DIGIBUILD/protocol/openid-connect/token"

    payload = f'grant_type=password&client_id=data_sharing&client_secret=20883f27-8f3c-4826-b908-c099b5ab279e&scope=openid&username={username}&password={password}'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        resp_json = response.json()
        return resp_json.get("access_token")
    except requests.RequestException as e:
        log.error(f"Error generating token: {e}")
        return None
    except json.JSONDecodeError as e:
        log.error(f"Error parsing token response: {e}")
        return None


def get_time_range(range_dict):
    """
    Calculates the time range based on the provided range dictionary.

    Parameters:
    - range_dict: dict, Range parameters with value and subject.

    Returns:
    - tuple, (time_from, time_to) as strings in format %Y%m%d%H%M.
    """
    if not range_dict:
        return None, None

    now = datetime.now()
    value = range_dict['value']
    subject = range_dict['subject']

    if subject == 'days':
        time_from = now - timedelta(days=value)
    elif subject == 'hours':
        time_from = now - timedelta(hours=value)
    elif subject == 'weeks':
        time_from = now - timedelta(weeks=value)
    elif subject == 'months':
        time_from = now - timedelta(weeks=value)
    elif subject == 'years':
        time_from = now - timedelta(weeks=value)
    else:
        time_from = None

    return int(time_from.strftime('%Y%m%d%H%M')), int(now.strftime('%Y%m%d%H%M'))


def construct_query(pilot, sensors, time_from=None, time_to=None, weather_locations=None):
    """
    Constructs the SQL query based on the provided parameters, including filtering by location_id if weather_locations are provided.

    Parameters:
    - pilot: str, The pilot name.
    - sensors: list, List of sensor IDs.
    - time_from: str, Optional start time for the query.
    - time_to: str, Optional end time for the query.
    - weather_locations: list, Optional list of location IDs for filtering.

    Returns:
    - str, Constructed SQL query.
    """
    if not sensors:
        return ''

    sensors_list = "', '".join(sensors)
    if weather_locations:
        base_query = (
            f"SELECT 	f_tsweatherdata.calendar_id, f_tsweatherdata.measurement_id, f_tsweatherdata.f_value, f_tsweatherdata.location_id "
            f"FROM {pilot}.public.f_tsweatherdata "
            f"WHERE f_tsweatherdata.measurement_id IN ('{sensors_list}')"
        )
    else:
        base_query = (
            f"SELECT f_tsdata.calendar_id, f_tsdata.sensor_id, f_tsdata.f_value, f_tsdata.location_id "
            f"FROM {pilot}.public.f_tsdata "
            f"WHERE f_tsdata.sensor_id IN ('{sensors_list}')"
        )

    if time_from and time_to:
        if weather_locations:
            base_query += f" AND f_tsweatherdata.calendar_id BETWEEN {time_to} AND {time_from}"
        else:
            base_query += f" AND f_tsdata.calendar_id BETWEEN {time_from} AND {time_to}"

    if weather_locations:
        locations_list = ",".join(weather_locations)
        base_query += f" AND f_tsweatherdata.location_id IN ({locations_list})"
    if weather_locations:
        base_query += " ORDER BY f_tsweatherdata.calendar_id ASC"
    else:
        base_query += " ORDER BY f_tsdata.calendar_id ASC"
    return base_query


def execute_query(token, pilot_db, sensors, time_from, time_to, weather_locations=None):
    """
    Executes the provided query for a specified data mart and converts the returned data into a structured pandas DataFrame.

    Parameters:
    - token: str, The authentication token for authorization.
    - pilot_db: str, The pilot database name.
    - sensors: list, List of sensor IDs.
    - time_from: str, Optional start time for the query.
    - time_to: str, Optional end time for the query.
    - weather_locations: list, Optional list of location IDs for filtering.

    Returns:
    - DataFrame, A pandas DataFrame containing the data for the specified sensors.
    """
    log.info(f"Execute query for sensors: {sensors}")
    url = "https://digibuild.epu.ntua.gr/data_sharing/federated_querying/execute_query/"

    query = construct_query(pilot_db, sensors, time_from, time_to, weather_locations)
    payload = json.dumps({"query": query})

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        resp_json = response.json()

        if not resp_json:
            log.warning("Empty response received.")
            return pd.DataFrame()

        df = pd.DataFrame(resp_json, columns=['datetime', 'sensor_id', 'value', 'location_id'])
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M')

        return df
    except requests.RequestException as e:
        log.error(f"Error executing query: {e}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        log.error(f"Error parsing query response: {e}")
        return pd.DataFrame()
    except KeyError as e:
        log.error(f"Unexpected response structure: {e}")
        return pd.DataFrame()


def process_in_batches(use_case, token):
    sensors = use_case.get('sensors', [])
    pilot_db = use_case.get('pilot_db', '')
    range_dict = use_case.get('range', None)
    resample_dict = use_case.get('resample', None)

    time_from, time_to = get_time_range(range_dict)

    batch_size = 5
    all_data = []

    for i in range(0, len(sensors), batch_size):
        batch_sensors = sensors[i:i+batch_size]
        df_batch = execute_query(token, pilot_db, batch_sensors, time_from, time_to)
        all_data.append(df_batch)

    if all_data:
        df_combined = pd.concat(all_data)

        df_pivot = df_combined.pivot_table(index='datetime', columns='sensor_id', values='value', aggfunc='mean')

        if resample_dict:
            resample_value = resample_dict['value']
            resample_subject = resample_dict['subject']
            resample_method = resample_dict['method']
            resample_rule = f"{resample_value}{resample_subject[0].upper()}"

            df_pivot = df_pivot.resample(resample_rule).apply(resample_method)

        # Flatten the columns and reset index
        df_pivot.columns = [f"{col}" for col in df_pivot.columns]
        df_pivot.reset_index(inplace=True)
        df_final = pd.DataFrame(df_pivot)

        return df_final
    else:
        return pd.DataFrame()

'''Here Provide your credentials'''

username = 'vcipollone'
password = '4K@y0<u6kVD('


'''Select the desired directory and load the data use cases'''
directory = os.getcwd()
use_cases = []
'''
use_cases = [
    {
        "id": 1,
        "description": "Testing the ML models for comfort classification",
        "sensors": ["134625", "144693", "144694", "144788", "144789", "144796", "144803", "144805", "144811", 
                    "144855", "144860", "144696", "144697", "144703", "144711", "144729", "145350", "145379", 
                    "149333", "1493345", "144856", "145080", "145156", "145272", "148934", "149060", "148830"],
        "resample": {
            "value": 10,
            "subject": "minutes",
            "method": "mean"
        },
        "pilot_db": "fvh_test"
    },
    {
        "id": 2,
        "description": "Testing the ML models for comfort prediction",
        "sensors": ["134625", "144693", "144694", "144788", "144789", "144796", "144803", "144805", "144811", 
                    "144855", "144860", "144696", "144697", "144703", "144711", "144729", "145350", "145379", 
                    "149333", "1493345", "144856", "145080", "145156", "145272", "148934", "149060", "148830"],
        "resample": {
            "value": 10,
            "subject": "minutes",
            "method": "mean"
        },
        "range": {
            "value": 1,
            "subject": "weeks"
        },
        "pilot_db": "fvh_test"
    },
    {
        "id": 3,
        "description": "Enenrgy forecasting - Training/testing model",
        "sensors": ["134625", "144693", "144694", "144788", "144789", "144796", "144803", "144805", "144811", 
                    "144855", "144860", "144696", "144697", "144703", "144711", "144729", "145350", "145379", 
                    "149333", "1493345", "144856", "145080", "145156", "145272", "148934", "149060", "148830"],
        "resample": {
            "value": 1,
            "subject": "hours",
            "method": "mean"
        },
        "range": {
            "value": 1,
            "subject": "weeks"
        },
        "pilot_db": "fvh_test"
    },
    {
        "id": 4,
        "description": "Optimization of energy consumption considering comfort mantainance - train",
        "sensors": ["134625", "144693", "144694", "144788", "144789", "144796", "144803", "144805", "144811", 
                    "144855", "144860", "144696", "144697", "144703", "144711", "144729", "145350", "145379", 
                    "149333", "1493345", "144856", "145080", "145156", "145272", "148934", "149060", "148830"],
        "resample": {
            "value": 1,
            "subject": "hours",
            "method": "mean"
        },
        "pilot_db": "fvh_test"
    }
    
]
'''

for filename in ['univpn_focchi-pilot.json']:
    file_path = os.path.join(directory, filename)
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
        use_cases.extend(data)

# Generate token
token = gen_token(username, password)

# Process each input
for use_case in use_cases:
    df = process_in_batches(use_case, token)
    print(df)



