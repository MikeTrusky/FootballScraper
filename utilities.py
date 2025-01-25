import configparser
import json
import time
import requests

#Premier League Id: 2021

URL_BEGIN           = "https://api.football-data.org/v4"
REQUESTS_DONE       = 0
REQUESTS_PER_MINUTE = 10
stats_file          = r'matchStats.json'
config_file         = r'config.cfg'

def get_api_key():
    config = configparser.RawConfigParser()
    config.read(config_file)
    return config.get("KEYS", "API_KEY")    

def get_headers_with_authToken():
    return { "X-Auth-Token": get_api_key() }

def save_to_file(results):
    with open(stats_file, "w") as file:
        json.dump(results, file, indent=4)

def load_from_file():
    with open(stats_file, "r") as file:
        return json.load(file)  
    
def do_request_with_limit_check(requestUrl):
    global REQUESTS_DONE

    if REQUESTS_DONE + 1 > REQUESTS_PER_MINUTE:    
        do_waiting_sleep()
        REQUESTS_DONE = 0

    REQUESTS_DONE += 1        
    response = requests.get(requestUrl, headers=get_headers_with_authToken())
    if response.status_code == 429: # Too many requests
        do_waiting_sleep()
        REQUESTS_DONE = 0
        response = requests.get(requestUrl, headers=get_headers_with_authToken())    
    
    return response

def do_waiting_sleep():
    print("[RequestsLimitCheck] .....waiting cause request limit....")
    time.sleep(60)
    print("[RequestsLimitCheck] Waiting is over")