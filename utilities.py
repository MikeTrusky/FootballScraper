import configparser
import json

#Premier League Id: 2021

URL_BEGIN       = "https://api.football-data.org/v4"
stats_file      = r'matchStats.json'
config_file     = r'config.cfg'

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