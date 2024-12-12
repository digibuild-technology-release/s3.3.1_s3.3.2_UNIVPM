import requests, json, logging, sys
import pandas as pd
import matplotlib.pyplot as plt


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
    url = "https://digibuild.epu.ntua.gr/data_sharing/federated_querying/execute_query/"
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


'''Here Provide your credentials'''

username = 'vcipollone'
password = '4K@y0<u6kVD('

#Define time range by adjusting time_from and time_to variables (fromat = %Y%m%d%H%M)
# date_from = '202303100000'
# date_to = '202405040000'

from datetime import datetime
#define pilot db
pilotdb = "fvh_test"

token = gen_token(username, password)
sensors=[  '144855','144856','144858','144860','144861','144863','144864','144865','144866','144867','144868','144870','144871','144873','144874','144875','144910','144911', '144912']
# sensorsco2_sezB=['144696','144858','144864','144865','144867','144873','144874']
# sensorstemp_sezB=['144855', '144860', '144863', '144866', '144868', '144870', '144875', '144910','144911','144912']
# sensorsrh_sezB=['144856', '144861','144871']
sensorsco2_sezB=['144858', '144864']
sensorstemp_sezB=['144855', '144860']
sensorsrh_sezB=['144856', '144861']
sensor_elect=['134625'] #, '134626', '134628', '134625' --> i datapoint per elettricita sono quelli di petteri no quelli di excel bu ntua
sensor_out=['144800','150017','150479','148678','148685','149800']

####DATAPONTS CONSUMPTION
hvac_el_cons=['150081', '150098', '150195', '150209', '150308', '150328', '150336', '150466']
tot_elec_floor1=['150110','150213' ,'150375'] #primo piano abc
tot_elec_floor2=['150114','150230' ,'150388'] #secondo piano abc
tot_elec_floor3=['150128','150251' ,'150403'] #terzo piano abc
tot_elec_floor4=['150143','150265' ,'150417'] #quarto piano abc
tot_elec_floor5=['150157','150279' ,'150431'] #quinto piano abc
tot_elec_floor6=['150171','150293' ,'150445'] #sesto piano abc
tot_elec_floor7=['150187','150306' ,'150461'] #settimo piano abc

qwerty
#update 20724 - petteri mi ha mandato i datapoint ma probabilmente non sono integrati nel datalake perche mi danno dei dataframe vuoti
df_elect=execute_query(token, pilotdb, tot_elec_floor7, time_from=None, time_to=None)
df_elect['data_type']='electricity'
df_elect['datetime']=pd.to_datetime(df_elect['datetime'].values, format='%Y%m%d%H%M%S')
df_elect = df_elect.sort_values(by='datetime')
df_elect=df_elect.reset_index(drop=True)

df_co2B=execute_query(token, pilotdb, sensorsco2_sezB, time_from=None, time_to=None)
df_co2B['data_type']='CO2 (ppm)'
df_co2B['datetime']=pd.to_datetime(df_co2B['datetime'].values, format='%Y%m%d%H%M%S')
df_co2B = df_co2B.sort_values(by='datetime')
df_co2B=df_co2B.reset_index(drop=True)

df_tempB=execute_query(token, pilotdb, sensorstemp_sezB, time_from=None, time_to=None)
df_tempB['data_type']='indoor T (°C)'
df_tempB['datetime']=pd.to_datetime(df_tempB['datetime'].values, format='%Y%m%d%H%M%S')
df_tempB = df_tempB.sort_values(by='datetime')
df_tempB=df_tempB.reset_index(drop=True)

df_rhB=execute_query(token, pilotdb, sensorsrh_sezB, time_from=None, time_to=None)
df_rhB['data_type']='indoor RH (%)'
df_rhB['datetime']=pd.to_datetime(df_rhB['datetime'].values, format='%Y%m%d%H%M%S')
df_rhB = df_rhB.sort_values(by='datetime')
df_rhB=df_rhB.reset_index(drop=True)

df_out=execute_query(token, pilotdb, sensor_out, time_from=None, time_to=None)
df_out['data_type']='outT'
df_out['datetime']=pd.to_datetime(df_out['datetime'].values, format='%Y%m%d%H%M%S')
df_out = df_out.sort_values(by='datetime')
df_out=df_out.reset_index(drop=True)



