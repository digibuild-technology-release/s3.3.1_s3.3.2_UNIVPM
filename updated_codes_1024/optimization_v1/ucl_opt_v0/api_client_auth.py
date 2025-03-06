import requests,logging,sys, json
# Constants
token_URL = "https://digibuild.epu.ntua.gr/auth/realms/DIGIBUILD/protocol/openid-connect/token"
BASE_URL = "https://services.digibuild-project.eu/prediction-api/"


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
    url = token_URL

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

def get_models(token, pilot_name):
    """
    Fetches model details for a given pilot from the 'get_models' API endpoint.
    
    Parameters:
        pilot_name (str): Name of the pilot project.
        token (str): Bearer token for authentication.
        
    Returns:
        dict: JSON response containing model details.
    """
    url = f"{BASE_URL}/get_models/"
    params = {'pilot_name': pilot_name} 
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error(f"Failed to fetch models: {e}")
        return {}

def get_info(token, pilot_name, variable, frequency):
    """
    Retrieves specific information for a given pilot, variable, and frequency from the 'get_info' API endpoint.
    
    Parameters:
        pilot_name (str): Name of the pilot project.
        variable (str): Variable of interest.
        frequency (str): Data collection frequency.
        token (str): Bearer token for authentication.
        
    Returns:
        dict: JSON response containing the requested information.
    """
    url = f"{BASE_URL}/get_info/"
    params = {'pilot_name': pilot_name, 'variable': variable, 'frequency': frequency}
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error(f"Failed to fetch info: {e}")
        return {}



def get_prediction(token, pilot_name, variable, frequency, identifier=None, algorithm=None):
    """
    Fetches prediction data for a specified pilot, variable, and frequency from the 'get_prediction' API endpoint.
    
    Parameters:
        pilot_name (str): Name of the pilot project.
        variable (str): Variable for which predictions are sought.
        frequency (str): Data collection frequency.
        token (str): Bearer token for authentication.
        identifier (str, optional): Unique identifier for data subset.
        algorithm (str, optional): Algorithm used for prediction.
        
    Returns:
        dict: JSON response containing prediction data.
    """
    url = f"{BASE_URL}/inference/get_prediction"
    params = {
        'pilot_name': pilot_name,
        'variable': variable,
        'frequency': frequency,
        'identifier': identifier,
        'algorithm': algorithm
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error(f"Failed to fetch prediction for pilot '{pilot_name}', variable '{variable}', frequency '{frequency}': {e}")
        return {}
    



def get_indicators(token, pilot_name, variable,  identifier=None):
    """
    Retrieves indicators for a specified pilot, variable, and frequency from the 'get_indicators' API endpoint.
    
    Parameters:
        pilot_name (str): Name of the pilot project.
        variable (str): Variable of interest.
        frequency (str): Data collection frequency.
        token (str): Bearer token for authentication.
        identifier (str, optional): Unique identifier for data subset.
        
    Returns:
        dict: JSON response containing the requested indicators.
    """
    url = f"{BASE_URL}/inference/get_indicators"
    params = {
        'pilot_name': pilot_name,
        'variable': variable,
        'identifier': identifier
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error(f"Failed to fetch indicators for pilot '{pilot_name}', variable '{variable}': {e}")
        return {}

if __name__ == "__main__":

    # username = ""
    # password = ""
    username = 'vcipollone'
    password = '4K@y0<u6kVD('
    
    token = gen_token(username, password)
    
    ########################
    # Example ucl
    ########################
    
    
    pilot_name1 = 'UCL'
    variable = 'energy_consumption'
    frequency = 'hourly'

    # Example API Calls
    print("Get Models:")
    model_retrieved1 = get_models(token, pilot_name1)
    print(model_retrieved1)

    print("\nGet Info:")
    info_retrieved1 = get_info(token, pilot_name1, variable, frequency)
    print(info_retrieved1)

    print("\nGet Prediction:")
    prediction_retrieved1 = get_prediction(token, pilot_name1, variable, frequency)
    print(prediction_retrieved1)


    ########################
    # Example FOCCHI
    ########################
    
    
    pilot_name5b = 'FOCCHI'
    variable = 'energy_consumption'
    frequency = 'hourly'

    # Example API Calls
    print("Get Models:")
    model_retrieved5b = get_models(token, pilot_name5b)
    print(model_retrieved5b)

    print("\nGet Info:")
    info_retrieved5b = get_info(token, pilot_name5b, variable, frequency)
    print(info_retrieved5b)

    print("\nGet Prediction:")
    prediction_retrieved5b = get_prediction(token, pilot_name5b, variable, frequency)
    print(prediction_retrieved5b)

    # print("\nGet Indicators")
    # prediction_retrieved5b = get_indicators(token, pilot_name5b, variable)
    # print(prediction_retrieved5b)

    # identifier = 'z1'
    # print("\nGet Prediction (specific zone/device):")
    # prediction_retrieved5b = get_prediction(token, pilot_name5b, variable, frequency, identifier)
    # print(prediction_retrieved5b)


    # print("\nGet Indicators (specific zone/device)")
    # prediction_retrieved5b = get_indicators(token, pilot_name5b, variable, identifier)
    # print(prediction_retrieved5b)


