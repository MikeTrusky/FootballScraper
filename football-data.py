import requests
from utilities import URL_BEGIN
from utilities import get_headers_with_authToken
import json

DATE_FROM = "2024-08-01"
DATE_TO = "2025-01-18"

# MATCH_ID = int(sys.argv[1])
# NUM_MATCHES = int(sys.argv[2])
# COMPETITION_ID = int(sys.argv[3])

MATCH_ID = 497620
NUM_MATCHES = 10
COMPETITION_ID = 2021

matches = []

def get_matches_by_teamId(teamId):
    return [
        match
        for match in matches
        if match["homeTeam"]["id"] == teamId or match["awayTeam"]["id"] == teamId
    ]

def get_team_results(teamId):    
    teamMatches = get_matches_by_teamId(teamId)    
    results = []    

    for index, match in enumerate(teamMatches[:NUM_MATCHES]):
        results.append(get_team_match_info(teamId, teamMatches[(index + 1):((index + 1) + 5)], teamMatches[index]))

    return results

def get_team_match_info(teamId, teamMatches, match):
    matchId = match["id"]    
    homeTeamId = match["homeTeam"]["id"]
    awayTeamId = match["awayTeam"]["id"]
    
    homeTeamValues = get_team_previousMatches_results(True, homeTeamId, get_team_previousMatches(teamId, homeTeamId, teamMatches, matchId))
    awayTeamValues = get_team_previousMatches_results(False, awayTeamId, get_team_previousMatches(teamId, awayTeamId, teamMatches, matchId))

    return {
        "matchId": matchId, 
        "matchHomeTeamValues": homeTeamValues,
        "matchAwayTeamValues": awayTeamValues,        
        "matchGoals": match["score"]["totalGoals"]
    }

def get_team_previousMatches(originalTeamId, currentTeamId, originalTeamMatches, matchId):
    if originalTeamId == currentTeamId:
        return originalTeamMatches
    else:
        teamMatches = get_matches_by_teamId(currentTeamId)
        index = next(i for i, match in enumerate(teamMatches) if match["id"] == matchId)
        return teamMatches[index + 1:NUM_MATCHES + index + 1]

def get_team_previousMatches_results(homeMatch, teamId, matches):
    goalsScoredAll = 0
    goalsScoredHomeOrAway = 0
    goalsConcededAll = 0
    goalsConcededHomeOrAway = 0
    matchesHome = 0
    matchesAway = 0

    for match in matches:
        if match["homeTeam"]["id"] == teamId:
            matchesHome += 1
            goalsScoredAll += match["score"]["home"]
            goalsConcededAll += match["score"]["away"]
            if homeMatch:
                goalsScoredHomeOrAway += match["score"]["home"]
                goalsConcededHomeOrAway += match["score"]["away"]
        else:
            matchesAway += 1
            goalsScoredAll += match["score"]["away"]
            goalsConcededAll += match["score"]["home"]
            if homeMatch == False:
                goalsScoredHomeOrAway += match["score"]["away"]
                goalsConcededHomeOrAway += match["score"]["home"]        

    goalsScoredAllAvg = round(goalsScoredAll / NUM_MATCHES, 2)
    goalsScoredHomeOrAwayAvg = round(goalsScoredHomeOrAway / matchesHome, 2)
    goalsConcededAllAvg = round(goalsConcededAll / NUM_MATCHES, 2)
    goalsConcededHomeOrAwayAvg = round(goalsConcededHomeOrAway / matchesHome, 2)
    
    if homeMatch:
        return {
            "goalsScoredAll": goalsScoredAll,
            "goalsScoredHome": goalsScoredHomeOrAway,
            "goalsConcededAll": goalsConcededAll,
            "goalsConcededHome": goalsConcededHomeOrAway,
            "goalsScoredAllAvg": goalsScoredAllAvg,
            "goalsScoredHomeAvg": goalsScoredHomeOrAwayAvg,
            "goalsConcededAllAvg": goalsConcededAllAvg,
            "goalsConcededHomeAvg": goalsConcededHomeOrAwayAvg,
        }
    else:
        return {
            "goalsScoredAll": goalsScoredAll,
            "goalsScoredAway": goalsScoredHomeOrAway,
            "goalsConcededAll": goalsConcededAll,
            "goalsConcededAway": goalsConcededHomeOrAway,
            "goalsScoredAllAvg": goalsScoredAllAvg,
            "goalsScoredAwayAvg": goalsScoredHomeOrAwayAvg,
            "goalsConcededAllAvg": goalsConcededAllAvg,
            "goalsConcededAwayAvg": goalsConcededHomeOrAwayAvg,
        }

url = f"{URL_BEGIN}/competitions/{COMPETITION_ID}/matches?dateFrom={DATE_FROM}&dateTo={DATE_TO}"
response = requests.get(url, headers=get_headers_with_authToken())
matchesFromRequest = response.json()['matches']
matches.clear()
for match in reversed(matchesFromRequest):
    matches.append({
        "id": match["id"],
        "homeTeam": {
            "id": match["homeTeam"]["id"],
            "name": match["homeTeam"]["name"]
        },
        "awayTeam": {
            "id": match["awayTeam"]["id"],
            "name": match["awayTeam"]["name"]
        },
        "score": {
            "home": match["score"]["fullTime"]["home"],
            "away": match["score"]["fullTime"]["away"],
            "totalGoals": match["score"]["fullTime"]["home"] + match["score"]["fullTime"]["away"]
        }
    })


url = f"{URL_BEGIN}/matches/{MATCH_ID}"
response = requests.get(url, headers=get_headers_with_authToken())
homeTeam = response.json()['homeTeam']
awayTeam = response.json()['awayTeam']

results = {}
results["matchesNumber"] = NUM_MATCHES
results["homeTeamName"] = homeTeam["name"]
results["awayTeamName"] = awayTeam["name"]
results["matchesHomeTeam"] = get_team_results(homeTeam["id"])
results["matchesAwayTeam"] = get_team_results(awayTeam["id"])

with open("matchStatsNew.json", "w") as file:
    json.dump(results, file, indent=4)