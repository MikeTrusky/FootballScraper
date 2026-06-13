import configparser
import json
import os
import time
import requests

#Premier League Id: 2021

URL_BEGIN           = "https://api.football-data.org/v4"
REQUESTS_DONE       = 0
REQUESTS_PER_MINUTE = 10
DEFAULT_STATS_FILE  = r'matchStats/matchStatsWesthamCrystal.json'
STATS_FILE_ENV_VAR  = 'FOOTBALL_STATS_FILE'
config_file         = r'config.cfg'

def get_stats_file():
    return os.environ.get(STATS_FILE_ENV_VAR, DEFAULT_STATS_FILE)

def get_api_key():
    config = configparser.RawConfigParser()
    config.read(config_file)
    return config.get("KEYS", "API_KEY")    

def get_headers_with_authToken():
    return { "X-Auth-Token": get_api_key() }

def save_to_file(results, path=None):
    target = path or get_stats_file()
    parent_dir = os.path.dirname(target)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(target, "w") as file:
        json.dump(results, file, indent=4)

def load_from_file(path=None):
    target = path or get_stats_file()
    with open(target, "r") as file:
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