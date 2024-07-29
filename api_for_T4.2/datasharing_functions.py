import requests, json, logging, sys
import pandas as pd
from dotenv import load_dotenv
import os
import urllib.parse

load_dotenv()


client_id=os.getenv("IDM_CLIENT_ID")
client_secret=os.getenv("IDM_CLIENT_SECRET")
realm=os.getenv("IDM_REALM")
base_url=os.getenv("IDM_BASE_URL")
datasharing_base_url=os.getenv("DATASHARING_BASE_URL")

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
    url = f"{base_url}/realms/{realm}/protocol/openid-connect/token"
    payload_parameters = {
        'grant_type': 'password',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'openid',
        'username': username,
        'password': password
    }

    payload = urllib.parse.urlencode(payload_parameters)
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    resp_json = json.loads(response.text)
    return resp_json["access_token"]

def execute_query(token, pilot, sensors, time_from = None, time_to = None):
    '''
        Purpose: To execute a provided query for a specified data mart and convert the returned data into a structured pandas DataFrame.
        Parameters:
            token: The authentication token for authorization.
            sensors: A list of the desired sensors to collect data
        Process:
            Logs the sensors list.
            Constructs a payload with the query and headers including the authorization token.
            Makes a POST request to the data sharing URL.
            Parses the JSON response and converts it into a pandas DataFrame.
        Returns: A pandas DataFrame containing the data for the specified sensors.
        '''
    log.info(f"Execute query for {sensors}")
    url = f"{datasharing_base_url}/data_sharing/federated_querying/execute_query/"
    if len(sensors) == 0:
        payload = json.dumps({
            "query": ''
        })
    else:
        base_query = f"SELECT calendar_id, sensor_id, f_value FROM {pilot}.public.f_tsdata WHERE (sensor_id = " + "'" + f"{sensors.pop(0)}" + "'"
        for sensor in sensors:
            base_query += " OR sensor_id = " + "'" + sensor + "'"
        base_query = base_query + ") "
        if time_to and time_from:
            base_query = base_query + f"AND calendar_id BETWEEN {time_from} AND {time_to}"
        elif time_to and not time_from:
            base_query += f"AND calendar_id <= {time_to}"
        elif not time_to and time_from:
            base_query += f"AND calendar_id >= {time_from}"
        payload = json.dumps({
            "query": base_query
        })
        print(base_query)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    resp_json = json.loads(response.text)
    df = pd.DataFrame(resp_json, columns=['datetime', 'sensor_id', 'value'])
    return df