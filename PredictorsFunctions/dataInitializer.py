from utilities import load_from_file
from sklearn.model_selection import train_test_split

import pandas as pd

def initialize_data():
    data_json = load_from_file()

    home_team_features = data_json['matchesHomeTeam'][0]
    away_team_features = data_json['matchesAwayTeam'][0]

    data = {
        'home_goals_avg':               [home_team_features['matchHomeTeamValues']['goalsScoredHomeAvg']],
        'home_conceded_avg':            [home_team_features['matchHomeTeamValues']['goalsConcededHomeAvg']],
        'hTeam_goals_all_avg':          [home_team_features['matchHomeTeamValues']['goalsScoredAllAvg']],
        'hTeam_conceded_all_avg':       [home_team_features['matchHomeTeamValues']['goalsConcededAllAvg']],
        'away_goals_avg':               [away_team_features['matchAwayTeamValues']['goalsScoredAwayAvg']],
        'away_conceded_avg':            [away_team_features['matchAwayTeamValues']['goalsConcededAwayAvg']],
        'aTeam_goals_all_avg':          [away_team_features['matchAwayTeamValues']['goalsScoredAllAvg']],
        'aTeam_conceded_all_avg':       [away_team_features['matchAwayTeamValues']['goalsConcededAllAvg']],
        'hTeam_rivals_rating_avg':      [home_team_features['matchHomeTeamValues']["rivalsRatingAvg"]],
        'aTeam_rivals_rating_avg':      [away_team_features['matchAwayTeamValues']["rivalsRatingAvg"]],
        'h2h_hTeam_goals_all_avg':      [home_team_features['head2headMatches']['matchHomeTeamValues']['goalsScoredAllAvg']],
        'h2h_hTeam_conceded_all_avg':   [home_team_features['head2headMatches']['matchHomeTeamValues']['goalsConcededAllAvg']],
        'h2h_hTeam_home_goals_avg':     [home_team_features['head2headMatches']['matchHomeTeamValues']['goalsScoredHomeAvg']],
        'h2h_hTeam_home_conceded_avg':  [home_team_features['head2headMatches']['matchHomeTeamValues']['goalsConcededHomeAvg']],
        'h2h_aTeam_goals_all_avg':      [away_team_features['head2headMatches']['matchAwayTeamValues']['goalsScoredAllAvg']],
        'h2h_aTeam_conceded_all_avg':   [away_team_features['head2headMatches']['matchAwayTeamValues']['goalsConcededAllAvg']],
        'h2h_aTeam_away_goals_avg':     [away_team_features['head2headMatches']['matchAwayTeamValues']['goalsScoredAwayAvg']],
        'h2h_aTeam_away_conceded_avg':  [away_team_features['head2headMatches']['matchAwayTeamValues']['goalsConcededAwayAvg']],
        'total_goals': [None]
    }

    for match in data_json['matchesHomeTeam'][1:] + data_json['matchesAwayTeam'][1:]:
        data['home_goals_avg'].append(match['matchHomeTeamValues']['goalsScoredHomeAvg'])
        data['home_conceded_avg'].append(match['matchHomeTeamValues']['goalsConcededHomeAvg'])
        data['hTeam_goals_all_avg'].append(match['matchHomeTeamValues']['goalsScoredAllAvg'])
        data['hTeam_conceded_all_avg'].append(match['matchHomeTeamValues']['goalsConcededAllAvg'])
        data['away_goals_avg'].append(match['matchAwayTeamValues']['goalsScoredAwayAvg'])
        data['away_conceded_avg'].append(match['matchAwayTeamValues']['goalsConcededAwayAvg'])
        data['aTeam_goals_all_avg'].append(match['matchAwayTeamValues']['goalsScoredAllAvg'])
        data['aTeam_conceded_all_avg'].append(match['matchAwayTeamValues']['goalsConcededAllAvg'])
        data['hTeam_rivals_rating_avg'].append(match['matchHomeTeamValues']["rivalsRatingAvg"])
        data['aTeam_rivals_rating_avg'].append(match['matchAwayTeamValues']["rivalsRatingAvg"])
        data['h2h_hTeam_goals_all_avg'].append(match['head2headMatches']['matchHomeTeamValues']['goalsScoredAllAvg'])
        data['h2h_hTeam_conceded_all_avg'].append(match['head2headMatches']['matchHomeTeamValues']['goalsConcededAllAvg'])
        data['h2h_hTeam_home_goals_avg'].append(match['head2headMatches']['matchHomeTeamValues']['goalsScoredHomeAvg'])
        data['h2h_hTeam_home_conceded_avg'].append(match['head2headMatches']['matchHomeTeamValues']['goalsConcededHomeAvg'])
        data['h2h_aTeam_goals_all_avg'].append(match['head2headMatches']['matchAwayTeamValues']['goalsScoredAllAvg'])
        data['h2h_aTeam_conceded_all_avg'].append(match['head2headMatches']['matchAwayTeamValues']['goalsConcededAllAvg'])
        data['h2h_aTeam_away_goals_avg'].append(match['head2headMatches']['matchAwayTeamValues']['goalsScoredAwayAvg'])
        data['h2h_aTeam_away_conceded_avg'].append(match['head2headMatches']['matchAwayTeamValues']['goalsConcededAwayAvg'])
        data['total_goals'].append(match['matchGoals'])

    return data

def prepare_dataFrames():
    data = initialize_data()
    df = pd.DataFrame(data)    

    train_df = df.dropna()    
    predict_df = df[df['total_goals'].isna()]    

    return train_df, predict_df

def get_available_features(train_df):
    available_features = [
        'home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg', 
        'hTeam_goals_all_avg', 'aTeam_goals_all_avg', 'hTeam_conceded_all_avg', 'aTeam_conceded_all_avg',
        'hTeam_rivals_rating_avg', 'aTeam_rivals_rating_avg', 'h2h_hTeam_goals_all_avg', 'h2h_hTeam_conceded_all_avg',
        'h2h_hTeam_home_goals_avg', 'h2h_hTeam_home_conceded_avg', 'h2h_aTeam_goals_all_avg', 'h2h_aTeam_conceded_all_avg',
        'h2h_aTeam_away_goals_avg', 'h2h_aTeam_away_conceded_avg'
    ]

    available_features = [col for col in available_features if col in train_df.columns]

    return available_features

def prepare_sets(train_df, predict_df, available_features):    
    X = train_df[available_features]    
    y = train_df['total_goals']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_predict = predict_df[available_features]

    # for column in X_train.columns:
    # print(f"{column}: mean={np.mean(X_train[column]):.2f}, std={np.std(X_train[column]):.2f}")

    return X_train, X_test, y_train, y_test, X_predict

def remove_redundant_features(train_df, predict_df, threshold=0.85):
    corr = train_df.corr()
    columns_to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                columns_to_drop.add(corr.columns[i])

    train_df = train_df.drop(columns=columns_to_drop)
    predict_df = predict_df.drop(columns=columns_to_drop)
    return train_df, predict_df