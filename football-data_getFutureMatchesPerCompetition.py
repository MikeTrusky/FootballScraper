#Get competitions' next matches between dates

import requests
import configparser
import sys

config = configparser.RawConfigParser()
config.read(r'config.cfg')

API_KEY         = config.get("KEYS", "API_KEY")
URL_BEGIN       = "https://api.football-data.org/v4"

COMPETITION_ID  = int(sys.argv[1])
DATE_FROM       = "2025-01-18"
DATE_TO         = "2025-01-20"

url = f"{URL_BEGIN}/competitions/{COMPETITION_ID}/matches?status=SCHEDULED&dateFrom={DATE_FROM}&dateTo={DATE_TO}"

headers = {
    "X-Auth-Token": API_KEY
}

response = requests.get(url, headers=headers)
matches = response.json()['matches']

matchesCollection = []

for match in matches:
    matchesCollection.append({
                                "matchId":  match["id"], 
                                "homeTeam": 
                                {
                                    "id":   match["homeTeam"]["id"], 
                                    "name": match["homeTeam"]["name"]
                                }, 
                                "awayTeam": 
                                {
                                    "id":   match["awayTeam"]["id"], 
                                    "name": match["awayTeam"]["name"]
                                }
                            })

print(matchesCollection)