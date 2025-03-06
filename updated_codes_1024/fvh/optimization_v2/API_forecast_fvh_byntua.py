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

    payload = f'grant_type=password&client_id=s3_2_3_heron&client_secret=8ae6b103-c18e-4fe3-bd49-045b39151cf6&scope=openid&username={username}&password={password}'
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

def get_data(token, section=None, floor=None, room=None):
    """
    Args:
    - section (str): Optional section 
    - floor (str): Optional floor 
    - room (str): Optional room 
    
    Returns:
    - response: response with predictions and datetime-value pairs.
    """
    url =  "https://digibuild.epu.ntua.gr/get_fvh_forecasts"
    
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    # Optional query parameters
    params = {}
    if section:
        params['section'] = section
    if floor:
        params['floor'] = floor
    if room:
        params['room'] = room

    # Make the GET request
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    # Return the API response
    return response.json()
