import requests
from utilities import get_api_key
from utilities import URL_BEGIN
import sys
import json

DATE_FROM = "2024-08-01"
DATE_TO = "2025-01-17"

MATCH_ID = int(sys.argv[1])
NUM_MATCHES = int(sys.argv[2])
COMPETITION_ID = int(sys.argv[3])

def get_team_info(homeMatch, teamId, teamName):
    goalsScoredAll = 0
    goalsScoredHomeOrAway = 0
    goalsConcededAll = 0
    goalsConcededHomeOrAway = 0
    matchesHome = 0
    matchesAway = 0

    url = f"{URL_BEGIN}/teams/{teamId}/matches?status=FINISHED&competitions={COMPETITION_ID}&dateFrom={DATE_FROM}&dateTo={DATE_TO}&limit={NUM_MATCHES}"

    headers = {
        "X-Auth-Token": get_api_key()
    }

    response = requests.get(url, headers=headers)
    matches = response.json()['matches']

    for match in matches:
        if match["homeTeam"]["id"] == teamId:
            matchesHome += 1
            goalsScoredAll += match["score"]["fullTime"]["home"]
            goalsConcededAll += match["score"]["fullTime"]["away"]
            if homeMatch:
                goalsScoredHomeOrAway += match["score"]["fullTime"]["home"]
                goalsConcededHomeOrAway += match["score"]["fullTime"]["away"]
        else:
            matchesAway += 1
            goalsScoredAll += match["score"]["fullTime"]["away"]
            goalsConcededAll += match["score"]["fullTime"]["home"]
            if homeMatch == False:
                goalsScoredHomeOrAway += match["score"]["fullTime"]["away"]
                goalsConcededHomeOrAway += match["score"]["fullTime"]["home"]        

    goalsScoredAllAvg = round(goalsScoredAll / NUM_MATCHES, 2)
    goalsScoredHomeOrAwayAvg = round(goalsScoredHomeOrAway / matchesHome, 2)
    goalsConcededAllAvg = round(goalsConcededAll / NUM_MATCHES, 2)
    goalsConcededHomeOrAwayAvg = round(goalsConcededHomeOrAway / matchesHome, 2)

    return {
        "name": teamName, 
        "goalsScoredAll": goalsScoredAll,
        "goalsScoredHomeOrAway": goalsScoredHomeOrAway,
        "goalsConcededAll": goalsConcededAll,
        "goalsConcededHomeOrAway": goalsConcededHomeOrAway,
        "goalsScoredAllAvg": goalsScoredAllAvg,
        "goalsScoredHomeOrAwayAvg": goalsScoredHomeOrAwayAvg,
        "goalsConcededAllAvg": goalsConcededAllAvg,
        "goalsConcededHomeOrAwayAvg": goalsConcededHomeOrAwayAvg
    }


url = f"{URL_BEGIN}/matches/{MATCH_ID}"

headers = {
    "X-Auth-Token": get_api_key()
}

response = requests.get(url, headers=headers)
homeTeam = response.json()['homeTeam']
awayTeam = response.json()['awayTeam']

result = {}
result["matchId"] = MATCH_ID
result["homeTeam"] = get_team_info(True, homeTeam["id"], homeTeam["name"])
result["awayTeam"] = get_team_info(False, awayTeam["id"], awayTeam["name"])
result["matchesNumber"] = NUM_MATCHES
#result["goalsAvgPredicted"] = 0

with open("matchStats.json", "w") as file:
    json.dump(result, file, indent=4)