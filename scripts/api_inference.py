import requests

import configparser

config = configparser.ConfigParser()

config.read('configs/config.ini')

SECRETS = config['SECRETS']


def query(payload):
	
    API_URL = SECRETS['api_url']
    
    headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {SECRETS['hf_token']}",
        "Content-Type": "application/json" 
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
	
    return response.json()