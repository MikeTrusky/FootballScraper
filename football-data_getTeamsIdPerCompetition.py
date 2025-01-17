#Get competitions' teams IDs

import requests
import configparser
import sys

config = configparser.RawConfigParser()
config.read(r'config.cfg')

API_KEY         = config.get("KEYS", "API_KEY")
URL_BEGIN       = "https://api.football-data.org/v4"

COMPETITION_ID  = int(sys.argv[1])

url = f"{URL_BEGIN}/competitions/{COMPETITION_ID}/teams"

headers = {
    "X-Auth-Token": API_KEY
}

response = requests.get(url, headers=headers)
teams = response.json()['teams']

teamsIds = []

for team in teams:
    teamsIds.append({
                        "name": team["name"], 
                        "id":   team["id"]
                    })    

print(teamsIds)