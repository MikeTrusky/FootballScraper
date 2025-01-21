import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#endregion

#region LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
y_pred = np.maximum(0, y_pred)

print("\nRegresja Liniowa:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f} \n")
#endregion

#region PolynomialRegression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

y_pred_poly = poly_reg.predict(X_test_poly)
y_pred_poly = np.maximum(0, y_pred_poly)

print("Regresja Wielomianowa (stopień 2):")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_poly):.2f} \n")
#endregion

#region DecisionTree
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train, y_train)

y_pred_tree = tree_reg.predict(X_test)
y_pred_tree = np.maximum(0, y_pred_tree)

print("Drzewo decyzyjne:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_tree):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_tree):.2f} \n")
#endregion

#region RandomForest
rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred_rf = rf_regressor.predict(X_test)

print("Random Forest:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f} \n")
#endregion

#region GradientBoosting
gb_regressor = GradientBoostingRegressor(random_state=42)
gb_regressor.fit(X_train, y_train)

y_pred_gb = gb_regressor.predict(X_test)
print("Gradient Boosting:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_gb):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_gb):.2f} \n")
#endregion

#region Prediction
X_predict = predict_df[['home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg',
                        'hTeam_goals_all_avg', 'aTeam_goals_all_avg', 'hTeam_conceded_all_avg', 'aTeam_conceded_all_avg']]

predicted_goals_linear = lin_reg.predict(X_predict)
print(f"Regresja liniowa: Przewidywana liczba goli w meczu: {predicted_goals_linear[0]:.2f}")

X_predict_poly = poly.transform(X_predict)
predicted_goals_polynomial = poly_reg.predict(X_predict_poly)
print(f"Regresja wielomianowa: Przewidywana liczba goli w meczu: {predicted_goals_polynomial[0]:.2f}")

predicted_goals_decisionTree = tree_reg.predict(X_predict)
print(f"Drzewo decyzyjne: Przewidywana liczba goli w meczu: {predicted_goals_decisionTree[0]:.2f}")

predicted_goals_rf = rf_regressor.predict(X_predict)
print(f"Random forest: Przewidywana liczba goli w meczu: {predicted_goals_rf[0]:.2f}")

predicted_goals_gb = gb_regressor.predict(X_predict)
print(f"Gradient Boosting: Przewidywana liczba goli w meczu: {predicted_goals_gb[0]:.2f} \n")
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
# plt.show()
#endregion