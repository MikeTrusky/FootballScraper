import configparser

#Premier League Id: 2021

URL_BEGIN       = "https://api.football-data.org/v4"
config_file     = r'config.cfg'

def get_api_key():
    config = configparser.RawConfigParser()
    config.read(config_file)
    return config.get("KEYS", "API_KEY")    

def get_headers_with_authToken():
    return { "X-Auth-Token": get_api_key() }