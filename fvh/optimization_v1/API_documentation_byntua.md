# FVH s3.1.1 API Documentation

The request can be done using an **empty body** to get the information for **all the building's spaces** or you can specify the spaces you want by the specifying the section, floor, room. You can specify any of these (eg. (section='A', floor='1') or (room='JKA1') etc.). The available spaces are the following:

<!--  -->

sections: A, B, C
floors: 0, 1, 2, 3, 4, 5, 6, 7
rooms: JKA0.1, JKA0.2, JKA1, JKA1.2_1, JKA1.2_2, JKA2,
        JKA2.1, JKA2.2, JKA3, JKA3.1, JKA3.2, JKA4, JKA4.1,
        JKA4.2, JKA5, JKA5.1, JKA5.2, JKA6, JKA6.1, JKA6.2,
        JKA7.1, JKA7.2, JKB0.1, JKB1, JKB1.1, JKB1.2, JKB1.3,
        JKB2, JKB2.1, JKB2.2, JKB2.3, JKB3, JKB3.1, JKB3.2,
        JKB4, JKB4.1, JKB4.2, JKB5, JKB5.1, JKB5.2, JKB6,
        JKB6.1, JKB6.2, JKB7.2, JKB7.3, JKC0.0, JKC0.1, JKC0.2,
        JKC0.3, JKC1, JKC1.1, JKC1.2, JKC2, JKC2.1, JKC2.2,
        JKC3, JKC3.1, JKC3.2, JKC4, JKC4.1, JKC4.2, JKC5,
        JKC5.1, JKC5.2, JKC6, JKC6.1, JKC6.2, JKC7.1, JKC7.2

An indicative request is presented below:

``` python
import requests

def get_data(section=None, floor=None, room=None):
    """
    Args:
    - section (str): Optional section 
    - floor (str): Optional floor 
    - room (str): Optional room 
    
    Returns:
    - response: response with predictions and datetime-value pairs.
    """
    url = "base_url+/get_fvh_forecasts"
    
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

```

The return response includes 24-hour forecasts for the specified spaces. In the example below (without params) the return response includes the forecasts for the entire buildingin total (Total), the sections in total (Sections), the floors in total (Floors) and each of the rooms (Rooms).


``` json
{
    "Total": {
        "Total": [
            {
                "2023-10-17 04:00:00+00:00": 42.529571533203125
            },
            {
                "2023-10-17 05:00:00+00:00": 50.10083770751953
            },
            ...
        ]
    },
    "Sections": {
        "A": [
            {
                "2023-10-17 04:00:00+00:00": 19.962186813354492
            },
            {
                "2023-10-17 05:00:00+00:00": 24.734485626220703
            },
            ...
        ],
        "B": [...],
        "C": [...]
    },
    "Floors": {
        "A_0": [...],
        "A_1": [...],
        ...
        "A_7": [...],
        "B_0": [...],
        ...
        "C_7": [...]
    },
    "Rooms": {
        "JKA0.1": [
            {
                "2023-10-17 04:00:00+00:00": 0.15318211913108826
            },
            {
                "2023-10-17 05:00:00+00:00": 0.16488000750541687
            },
            ...       
        ]
    }
}
```