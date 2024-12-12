# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:45:10 2024

@author: utente
"""

import requests
import json


BASE_URL="https://digibuild.epu.ntua.gr/"
username = 'vcipollone'
password = '4K@y0<u6kVD('
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


def get_data(section=None, floor=None, room=None):
    """
    Args:
    - section (str): Optional section 
    - floor (str): Optional floor 
    - room (str): Optional room 
    
    Returns:
    - response: response with predictions and datetime-value pairs.
    """
    url =  "https://digibuild.epu.ntua.gr/get_fvh_forecasts"
    
    # Optional query parameters
    params = {}
    if section:
        params['section'] = section
    if floor:
        params['floor'] = floor
    if room:
        params['room'] = room

    # Make the GET request
    response = requests.get(url, params=params)
    
    # Return the API response
    return response.json()

# Example usage
token = gen_token(username, password)

# To get all the rooms in section A:
response = get_data(section='A')
# To get all the rooms in floor 2:
response = get_data(floor='2')
# To get all the rooms in section A, floor 2:
response = get_data(section='A', floor='2')
# To get only get the room JKA1:
response = get_data(room='JKAI')
# This will also only give you the room JKA1:
response = get_data(section='A', floor='1', room='JKAI')

# But this will not work (room JKA1 is in section A, floor 1)!
response = get_data(section='B', floor='2', room='JKAI')