# import requests
from utilities import URL_BEGIN
from utilities import save_to_file
from utilities import do_request_with_limit_check
import math
import sys

DATE_FROM = "2024-08-01"
DATE_TO = "2025-02-01"

MATCH_ID = int(sys.argv[1])
# NUM_MATCHES = int(sys.argv[2])
# COMPETITION_ID = int(sys.argv[3])

# MATCH_ID = 497623
NUM_MATCHES = 5
COMPETITION_ID = 2021
SEASON = 2024
NUM_H2H_MATCHES = 6

matches = []
teamsRaiting = {}

#region Methods
def get_matches_by_teamId(teamId, currentMatchday):
    return [
        match
        for match in matches
        if (match["homeTeam"]["id"] == teamId or match["awayTeam"]["id"] == teamId)                
        and (match["matchday"] <= currentMatchday)
    ]

def get_team_results(teamId, matchday):    
    teamMatches = get_matches_by_teamId(teamId, matchday)    
    results = []    

    for index, match in enumerate(teamMatches[:NUM_MATCHES]):
        results.append(get_team_match_info(teamId, teamMatches[(index + 1):((index + 1) + NUM_MATCHES)], teamMatches[index], matchday))

    return results

def get_team_match_info(teamId, teamMatches, match, matchday):
    matchId = match["id"]    
    homeTeamId = match["homeTeam"]["id"]
    awayTeamId = match["awayTeam"]["id"]
    
    homeTeamValues = get_team_previousMatches_results(True, homeTeamId, get_team_previousMatches(teamId, homeTeamId, teamMatches, matchId, matchday), False)
    awayTeamValues = get_team_previousMatches_results(False, awayTeamId, get_team_previousMatches(teamId, awayTeamId, teamMatches, matchId, matchday), False)

    h2hMatches = get_head2head_matches(matchId)

    h2h_homeTeamValues = get_team_previousMatches_results(True, homeTeamId, h2hMatches, True)
    h2h_awayTeamValues = get_team_previousMatches_results(False, awayTeamId, h2hMatches, True)

    return {
        "matchId": matchId, 
        "homeTeamName": match["homeTeam"]["name"],
        "awayTeamName": match["awayTeam"]["name"],
        "matchHomeTeamValues": homeTeamValues,
        "matchAwayTeamValues": awayTeamValues,
        "h2hMatches": h2hMatches,
        "head2headMatches": {
            "matchHomeTeamValues": h2h_homeTeamValues,
            "matchAwayTeamValues": h2h_awayTeamValues,
        },
        "rivalRatingDifference": round(homeTeamValues["rivalsRatingAvg"] - awayTeamValues["rivalsRatingAvg"], 2),
        "matchGoals": match["score"]["totalGoals"]
    }

def get_team_previousMatches(originalTeamId, currentTeamId, originalTeamMatches, matchId, matchday):
    if originalTeamId == currentTeamId:
        return originalTeamMatches
    else:
        teamMatches = get_matches_by_teamId(currentTeamId, matchday)
        index = next(i for i, match in enumerate(teamMatches) if match["id"] == matchId)
        return teamMatches[index + 1:NUM_MATCHES + index + 1]

def get_team_previousMatches_results(homeMatch, teamId, methodMatches, h2h):
    goalsScoredAll = 0
    goalsScoredHomeOrAway = 0
    goalsConcededAll = 0
    goalsConcededHomeOrAway = 0
    matchesToDivide = 0    
    rivalsRating = 0

    for match in methodMatches:
        if match["homeTeam"]["id"] == teamId:
            rivalsRating += teamsRaiting[match["awayTeam"]["id"]]            
            goalsScoredAll += match["score"]["home"]
            goalsConcededAll += match["score"]["away"]
            if homeMatch:
                matchesToDivide += 1
                goalsScoredHomeOrAway += match["score"]["home"]
                goalsConcededHomeOrAway += match["score"]["away"]
        else:
            rivalsRating += teamsRaiting[match["homeTeam"]["id"]]            
            goalsScoredAll += match["score"]["away"]
            goalsConcededAll += match["score"]["home"]
            if homeMatch == False:
                matchesToDivide += 1
                goalsScoredHomeOrAway += match["score"]["away"]
                goalsConcededHomeOrAway += match["score"]["home"]            

    if h2h:
        rivalsRating = 0

    goalsScoredAllAvg = round(goalsScoredAll / NUM_MATCHES, 2)
    goalsScoredHomeOrAwayAvg = 0 if matchesToDivide <= 0 else round(goalsScoredHomeOrAway / matchesToDivide, 2)
    goalsConcededAllAvg = round(goalsConcededAll / NUM_MATCHES, 2)
    goalsConcededHomeOrAwayAvg = 0 if matchesToDivide <= 0 else round(goalsConcededHomeOrAway / matchesToDivide, 2)
    rivalsRatingAvg = round(rivalsRating / NUM_MATCHES, 2)
    
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
            "rivalsRatingAvg": rivalsRatingAvg,
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
            "rivalsRatingAvg": rivalsRatingAvg,
        }
    
def set_teamRating_for_matchday(matchday):
    standingsUrl = f"{URL_BEGIN}/competitions/{COMPETITION_ID}/standings?season={SEASON}&matchday={matchday - 1}"    
    standingsResponse = do_request_with_limit_check(standingsUrl)
    standingsTable = standingsResponse.json()['standings']

    total_table = next((s['table'] for s in standingsTable if s['type'] == 'TOTAL'), [])
    return { entry['team']['id']: entry['position'] for entry in total_table}

def get_matches_from_request(requestUrl):        
    response = do_request_with_limit_check(requestUrl)
    matchesFromRequest = response.json()['matches']
    matches.clear()
    for match in reversed(matchesFromRequest):        
        matches.append({
            "id": match["id"],
            "matchday": match["matchday"],
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
                "totalGoals": (match["score"]["fullTime"]["home"] if match["score"]["fullTime"]["home"] is not None else 0)
                             + (match["score"]["fullTime"]["away"] if match["score"]["fullTime"]["away"] is not None else 0)
            }
        })

def get_match_info():
    response = do_request_with_limit_check(f"{URL_BEGIN}/matches/{MATCH_ID}")
    homeTeam = response.json()['homeTeam']
    awayTeam = response.json()['awayTeam']
    matchMatchday = int(response.json()['matchday'])    

    return {
        "homeTeam": homeTeam,
        "awayTeam": awayTeam,
        "matchday": matchMatchday
    }

def update_raiting_by_previous_standings(matchday):
    if (matchday - NUM_MATCHES) > 0:
        teamsRaitingPrevious = set_teamRating_for_matchday(matchday - NUM_MATCHES)
        
        for team_id, current_rating in teamsRaiting.items():
            previous_rating = teamsRaitingPrevious.get(team_id)
            if previous_rating:
                teamsRaiting[team_id] = math.floor((current_rating + previous_rating) / 2) 

def set_results(homeTeam, awayTeam, matchday):
    results = {}
    results["matchesNumber"] = NUM_MATCHES
    results["homeTeamName"] = homeTeam["name"]
    results["awayTeamName"] = awayTeam["name"]
    results["matchesHomeTeam"] = get_team_results(homeTeam["id"], matchday)
    results["matchesAwayTeam"] = get_team_results(awayTeam["id"], matchday)    

    return results

def get_head2head_matches(matchId):    
    response = do_request_with_limit_check(f"{URL_BEGIN}/matches/{matchId}/head2head?limit={NUM_H2H_MATCHES}")    
    matchesFromRequest = response.json()['matches']        
    head2head_matches = []    
    for match in matchesFromRequest:
        if match["id"] != matchId:
            head2head_matches.append({
                "id": match["id"],
                "matchday": match["matchday"],
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
    return head2head_matches

#endregion

get_matches_from_request(f"{URL_BEGIN}/competitions/{COMPETITION_ID}/matches?dateFrom={DATE_FROM}&dateTo={DATE_TO}")
match_info = get_match_info()
teamsRaiting = set_teamRating_for_matchday(match_info["matchday"] - 1)
update_raiting_by_previous_standings(match_info["matchday"])
results = set_results(match_info["homeTeam"], match_info["awayTeam"], match_info["matchday"])

save_to_file(results)