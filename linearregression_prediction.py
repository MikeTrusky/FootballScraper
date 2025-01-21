import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

with open("matchStatsNew.json", "r") as file:
    data_json = json.load(file)

#region DataInitialization
home_team_features = data_json['matchesHomeTeam'][0]
away_team_features = data_json['matchesAwayTeam'][0]

data = {
    'home_goals_avg': [home_team_features['matchHomeTeamValues']['goalsScoredHomeAvg']],
    'home_conceded_avg': [home_team_features['matchHomeTeamValues']['goalsConcededHomeAvg']],
    'hTeam_goals_all_avg': [home_team_features['matchHomeTeamValues']['goalsScoredAllAvg']],
    'hTeam_conceded_all_avg': [home_team_features['matchHomeTeamValues']['goalsConcededAllAvg']],
    'away_goals_avg': [away_team_features['matchAwayTeamValues']['goalsScoredAwayAvg']],
    'away_conceded_avg': [away_team_features['matchAwayTeamValues']['goalsConcededAwayAvg']],
    'aTeam_goals_all_avg': [away_team_features['matchAwayTeamValues']['goalsScoredAllAvg']],
    'aTeam_conceded_all_avg': [away_team_features['matchAwayTeamValues']['goalsConcededAllAvg']],
    'total_goals': [None]
}

for match in data_json['matchesHomeTeam'][1:]:
    data['home_goals_avg'].append(match['matchHomeTeamValues']['goalsScoredHomeAvg'])
    data['home_conceded_avg'].append(match['matchHomeTeamValues']['goalsConcededHomeAvg'])
    data['hTeam_goals_all_avg'].append(match['matchHomeTeamValues']['goalsScoredAllAvg'])
    data['hTeam_conceded_all_avg'].append(match['matchHomeTeamValues']['goalsConcededAllAvg'])
    data['away_goals_avg'].append(match['matchAwayTeamValues']['goalsScoredAwayAvg'])
    data['away_conceded_avg'].append(match['matchAwayTeamValues']['goalsConcededAwayAvg'])
    data['aTeam_goals_all_avg'].append(match['matchAwayTeamValues']['goalsScoredAllAvg'])
    data['aTeam_conceded_all_avg'].append(match['matchAwayTeamValues']['goalsConcededAllAvg'])
    data['total_goals'].append(match['matchGoals'])

for match in data_json['matchesAwayTeam'][1:]:
    data['home_goals_avg'].append(match['matchHomeTeamValues']['goalsScoredHomeAvg'])
    data['home_conceded_avg'].append(match['matchHomeTeamValues']['goalsConcededHomeAvg'])
    data['hTeam_goals_all_avg'].append(match['matchHomeTeamValues']['goalsScoredAllAvg'])
    data['hTeam_conceded_all_avg'].append(match['matchHomeTeamValues']['goalsConcededAllAvg'])
    data['away_goals_avg'].append(match['matchAwayTeamValues']['goalsScoredAwayAvg'])
    data['away_conceded_avg'].append(match['matchAwayTeamValues']['goalsConcededAwayAvg'])
    data['aTeam_goals_all_avg'].append(match['matchAwayTeamValues']['goalsScoredAllAvg'])
    data['aTeam_conceded_all_avg'].append(match['matchAwayTeamValues']['goalsConcededAllAvg'])
    data['total_goals'].append(match['matchGoals'])
#endregion

#region SetDfCollections
df = pd.DataFrame(data)

train_df = df.dropna()
predict_df = df[df['total_goals'].isna()]

# print("Dane treningowe:")
# print(train_df.describe())
# print(train_df.isna().sum())  # Sprawdzi brakujące wartości
#endregion

#region SetXy
X = train_df[
    [
        'home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg', 
        'hTeam_goals_all_avg', 'aTeam_goals_all_avg', 'hTeam_conceded_all_avg', 'aTeam_conceded_all_avg'
    ]
]
y = train_df['total_goals']
#endregion

#region LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
y_pred = np.maximum(0, y_pred)

print("Regresja Liniowa:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
#endregion

#region PolynomialRegression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

y_pred_poly = poly_reg.predict(X_test_poly)
y_pred_poly = np.maximum(0, y_pred_poly)

print("\nRegresja Wielomianowa (stopień 2):")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_poly):.2f}")
#endregion

#region Prediction
X_predict = predict_df[['home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg',
                        'hTeam_goals_all_avg', 'aTeam_goals_all_avg', 'hTeam_conceded_all_avg', 'aTeam_conceded_all_avg']]

predicted_goals = lin_reg.predict(X_predict)
predicted_goals = np.maximum(0, predicted_goals)
print(f"Przewidywana liczba goli w meczu: {predicted_goals[0]:.2f}")
#endregion

#region Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, label='Regresja Liniowa', alpha=0.7)
plt.scatter(y_test, y_pred_poly, label='Regresja Wielomianowa', alpha=0.7, color='r')
plt.plot([0, 6], [0, 6], '--', color='gray')  # Idealna linia
plt.xlabel("Rzeczywista liczba goli")
plt.ylabel("Przewidywana liczba goli")
plt.legend()
plt.title("Porównanie modeli regresji")
plt.show()
#endregion