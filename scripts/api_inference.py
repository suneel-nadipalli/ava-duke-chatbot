import requests

import configparser

config = configparser.ConfigParser()

config.read('configs/config.ini')

SECRETS = config['SECRETS']

import sys

sys.path.append('..')


def query(payload):
    """
    Purpose: Query the model with the payload.
    Input: payload - The payload to query the model with.
    """
	
    API_URL = SECRETS['api_url']

    # The headers contain the authorization token and the content type
    
    headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {SECRETS['hf_token']}",
        "Content-Type": "application/json" 
    }
    
    # The response is the result of the post request to the API URL with the headers and the payload
    
    response = requests.post(API_URL, headers=headers, json=payload)
	
    return response.json()