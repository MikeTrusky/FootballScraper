#Get competitions' next matches between dates

import requests
from utilities import get_api_key
from utilities import URL_BEGIN
import sys

COMPETITION_ID  = int(sys.argv[1])
DATE_FROM       = "2025-01-18"
DATE_TO         = "2025-01-20"

url = f"{URL_BEGIN}/competitions/{COMPETITION_ID}/matches?status=SCHEDULED&dateFrom={DATE_FROM}&dateTo={DATE_TO}"

headers = {
    "X-Auth-Token": get_api_key()
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