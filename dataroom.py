import requests, json, logging, sys
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Ottieni username, password e URL da variabili d'ambiente
username = os.getenv("KEYCLOACK_USERNAME")
password = os.getenv("KEYCLOACK_PASSWORD")
keycloak_url = os.getenv("KEYCLOACK_URL")
url_data = os.getenv("URL_DATA")

# Funzione per generare il token
def gen_token(username, password):
    payload = f'grant_type=password&client_id=data_sharing&client_secret=20883f27-8f3c-4826-b908-c099b5ab279e&scope=openid&username={username}&password={password}'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    try:
        response = requests.post(keycloak_url, headers=headers, data=payload)
        response.raise_for_status()
        resp_json = response.json()
        return resp_json["access_token"]
    except requests.exceptions.RequestException as e:
        log.error(f"Error while getting token: {str(e)}")
        return None


def get_time_range(range_dict):
    """
    Calculates the time range based on the provided range dictionary.
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
    else:
        time_from = None

    return int(time_from.strftime('%Y%m%d%H%M')), int(now.strftime('%Y%m%d%H%M'))


def construct_query(pilot, sensors, time_from=None, time_to=None):
    """
    Constructs the SQL query based on the provided parameters.
    """
    if not sensors:
        return ''

    # Assicurati che ogni sensore sia una stringa
    sensors_list = "', '".join([str(sensor) for sensor in sensors])  # Converti i sensori in stringhe
    base_query = (
        f"SELECT f_tsdata.calendar_id, f_tsdata.sensor_id, f_tsdata.f_value "
        f"FROM {pilot}.public.f_tsdata "
        f"WHERE f_tsdata.sensor_id IN ('{sensors_list}')"
    )

    if time_from and time_to:
        base_query += f" AND f_tsdata.calendar_id BETWEEN {time_from} AND {time_to}"

    base_query += " ORDER BY f_tsdata.calendar_id ASC"
    return base_query



def execute_query(token, pilot_db, sensors, time_from, time_to):
    """
    Executes the provided query and converts the returned data into a structured pandas DataFrame.
    """
    log.info(f"Execute query for sensors: {sensors}")
    url = url_data

    query = construct_query(pilot_db, sensors, time_from, time_to)
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

        df = pd.DataFrame(resp_json, columns=['datetime', 'sensor_id', 'value'])
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


def process_in_batches(use_case, token, df_sensors):
    sensors = use_case.get('sensors', [])
    pilot_db = use_case.get('pilot_db', '')
    range_dict = use_case.get('range', None)

    time_from, time_to = get_time_range(range_dict)

    batch_size = 600
    all_data = []

    for i in range(0, len(sensors), batch_size):
        batch_sensors = [str(sensor) for sensor in sensors[i:i+batch_size]]  # Assicurati che i sensori siano stringhe
        df_batch = execute_query(token, pilot_db, batch_sensors, time_from, time_to)
        all_data.append(df_batch)

    if all_data:
        df_combined = pd.concat(all_data)
        df_pivot = df_combined.pivot_table(index='datetime', columns='sensor_id', values='value', aggfunc='mean')

        df_sensors['sensor_id'] = df_sensors['sensor_id'].astype(str).str.strip()  # Converti e rimuovi spazi
        sensor_mapping = df_sensors.set_index('sensor_id')['sensor_type'].to_dict()
        df_pivot.rename(columns={col: f"{col}_{sensor_mapping.get(str(col), 'Unknown')}" for col in df_pivot.columns}, inplace=True)

        df_pivot.reset_index(inplace=True)
        df_final = pd.DataFrame(df_pivot)

        return df_final
    else:
        return pd.DataFrame()


# 🔥 AGGIUNTA DELLE FUNZIONI PER CREARE UN FILE EXCEL PER OGNI STANZA 🔥

def load_sensors_from_excel(file_path):
    """Carica il file Excel e restituisce un DataFrame."""
    return pd.read_excel(file_path)

def get_unique_rooms(df):
    """Restituisce la lista unica delle stanze presenti nel file."""
    return df["room_id"].unique()

def get_sensor_ids(df, room_id):
    """Restituisce tutti i sensori associati a una stanza."""
    return df[df["room_id"] == room_id]["sensor_id"].tolist()