#Get competitions' teams IDs

import requests
from utilities import get_api_key
from utilities import URL_BEGIN
import sys

COMPETITION_ID  = int(sys.argv[1])

url = f"{URL_BEGIN}/competitions/{COMPETITION_ID}/teams"

headers = {
    "X-Auth-Token": get_api_key()
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