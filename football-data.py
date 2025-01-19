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

def get_team_results(homeMatch, teamId):
    url = f"{URL_BEGIN}/teams/{teamId}/matches?status=FINISHED&competitions={COMPETITION_ID}&dateFrom={DATE_FROM}&dateTo={DATE_TO}&limit={NUM_MATCHES * 2}"

    headers = {
        "X-Auth-Token": get_api_key()
    }

    response = requests.get(url, headers=headers)
    matches = response.json()['matches']

    results = []
    matches.reverse()
    for index, match in enumerate(matches[:NUM_MATCHES]):
        if index == 0:
            results.append(get_team_match_info(homeMatch, teamId, 0, matches[index:(index + 5)], 0))
        else:
            results.append(get_team_match_info(homeMatch, teamId, matches[(index - 1)]["id"], matches[index:(index + 5)], matches[(index - 1)]["score"]["fullTime"]["home"] + matches[(index - 1)]["score"]["fullTime"]["away"]))

    return results

def get_team_match_info(homeMatch, teamId, matchId, matches, matchTotalGoals):
    goalsScoredAll = 0
    goalsScoredHomeOrAway = 0
    goalsConcededAll = 0
    goalsConcededHomeOrAway = 0
    matchesHome = 0
    matchesAway = 0

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
        "matchId": matchId, 
        "goalsScoredAll": goalsScoredAll,
        "goalsScoredHomeOrAway": goalsScoredHomeOrAway,
        "goalsConcededAll": goalsConcededAll,
        "goalsConcededHomeOrAway": goalsConcededHomeOrAway,
        "goalsScoredAllAvg": goalsScoredAllAvg,
        "goalsScoredHomeOrAwayAvg": goalsScoredHomeOrAwayAvg,
        "goalsConcededAllAvg": goalsConcededAllAvg,
        "goalsConcededHomeOrAwayAvg": goalsConcededHomeOrAwayAvg,
        "matchGoals": matchTotalGoals
    }


url = f"{URL_BEGIN}/matches/{MATCH_ID}"

headers = {
    "X-Auth-Token": get_api_key()
}

response = requests.get(url, headers=headers)
homeTeam = response.json()['homeTeam']
awayTeam = response.json()['awayTeam']

results = {}
results["matchesNumber"] = NUM_MATCHES
results["homeTeamName"] = homeTeam["name"]
results["awayTeamName"] = awayTeam["name"]
results["matchesHomeTeam"] = get_team_results(True, homeTeam["id"])
results["matchesAwayTeam"] = get_team_results(False, awayTeam["id"])

with open("matchStatsNew.json", "w") as file:
    json.dump(results, file, indent=4)