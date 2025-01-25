import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from utilities import load_from_file

def calculate_cross_score(model, name):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_mse = -scores.mean()
    print(f"Cross-validated MSE for {name}: {mean_mse:.2f}")

data_json = load_from_file()

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
    'hTeam_rivals_rating_avg': [home_team_features['matchHomeTeamValues']["rivalsRatingAvg"]],
    'aTeam_rivals_rating_avg': [away_team_features['matchAwayTeamValues']["rivalsRatingAvg"]],
    'h2h_hTeam_goals_all_avg': [home_team_features['head2headMatches']['matchHomeTeamValues']['goalsScoredAllAvg']],
    'h2h_hTeam_conceded_all_avg': [home_team_features['head2headMatches']['matchHomeTeamValues']['goalsConcededAllAvg']],
    'h2h_hTeam_home_goals_avg': [home_team_features['head2headMatches']['matchHomeTeamValues']['goalsScoredHomeAvg']],
    'h2h_hTeam_home_conceded_avg': [home_team_features['head2headMatches']['matchHomeTeamValues']['goalsConcededHomeAvg']],
    'h2h_aTeam_goals_all_avg': [away_team_features['head2headMatches']['matchAwayTeamValues']['goalsScoredAllAvg']],
    'h2h_aTeam_conceded_all_avg': [away_team_features['head2headMatches']['matchAwayTeamValues']['goalsConcededAllAvg']],
    'h2h_aTeam_away_goals_avg': [away_team_features['head2headMatches']['matchAwayTeamValues']['goalsScoredAwayAvg']],
    'h2h_aTeam_away_conceded_avg': [away_team_features['head2headMatches']['matchAwayTeamValues']['goalsConcededAwayAvg']],
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

for match in data_json['matchesAwayTeam'][1:]:
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
        'hTeam_goals_all_avg', 'aTeam_goals_all_avg', 'hTeam_conceded_all_avg', 'aTeam_conceded_all_avg',
        'hTeam_rivals_rating_avg', 'aTeam_rivals_rating_avg', 'h2h_hTeam_goals_all_avg', 'h2h_hTeam_conceded_all_avg',
        'h2h_hTeam_home_goals_avg', 'h2h_hTeam_home_conceded_avg', 'h2h_aTeam_goals_all_avg', 'h2h_aTeam_conceded_all_avg',
        'h2h_aTeam_away_goals_avg', 'h2h_aTeam_away_conceded_avg'
    ]
]
y = train_df['total_goals']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#endregion

#region Naive
naive_pred = np.mean(y_train)
naive_mse = np.mean((y_test - naive_pred) ** 2)
print(f"Naive Baseline MSE: {naive_mse}")
#endregion

#region LinearRegression
lin_reg = LinearRegression()
# calculate_cross_score(lin_reg, "Linear")
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
# calculate_cross_score(tree_reg, "DecisionTree")
tree_reg.fit(X_train, y_train)

y_pred_tree = tree_reg.predict(X_test)
y_pred_tree = np.maximum(0, y_pred_tree)

print("Drzewo decyzyjne:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_tree):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_tree):.2f} \n")
#endregion

#region RandomForest
rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
# calculate_cross_score(rf_regressor, "RandomForest")
rf_regressor.fit(X_train, y_train)

y_pred_rf = rf_regressor.predict(X_test)

print("Random Forest:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f} \n")
#endregion

#region GradientBoosting
gb_regressor = GradientBoostingRegressor(random_state=42)
# calculate_cross_score(gb_regressor, "Gradient")
gb_regressor.fit(X_train, y_train)

y_pred_gb = gb_regressor.predict(X_test)
print("Gradient Boosting:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_gb):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_gb):.2f} \n")
#endregion

#region XGBRegressor
xgb_reg = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
# calculate_cross_score(xgb_reg, "XGBRegressor")
xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
# y_pred_xgb = np.maximum(0, y_pred_xgb)

print("XGBoost:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_xgb):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_xgb):.2f} \n")
#endregion

#region NeuralNetwork
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='relu')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=4, verbose=0)

y_pred_nn = model.predict(X_test).flatten()

print("Neural network:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_nn):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_nn):.2f} \n")
#endregion

#region Prediction
X_predict = predict_df[['home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg',
                        'hTeam_goals_all_avg', 'aTeam_goals_all_avg', 'hTeam_conceded_all_avg', 'aTeam_conceded_all_avg',
                        'hTeam_rivals_rating_avg', 'aTeam_rivals_rating_avg',  'h2h_hTeam_goals_all_avg', 'h2h_hTeam_conceded_all_avg',
                        'h2h_hTeam_home_goals_avg', 'h2h_hTeam_home_conceded_avg', 'h2h_aTeam_goals_all_avg', 'h2h_aTeam_conceded_all_avg',
                        'h2h_aTeam_away_goals_avg', 'h2h_aTeam_away_conceded_avg']]

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
print(f"Gradient Boosting: Przewidywana liczba goli w meczu: {predicted_goals_gb[0]:.2f}")

predicted_goals_xgb = xgb_reg.predict(X_predict)
print(f"XGBoost: Przewidywana liczba goli w meczu: {predicted_goals_xgb[0]:.2f}")

predicted_goals_nn = model.predict(X_predict, verbose=0).flatten()
print(f"Neural: Przewidywana liczba goli w meczu: {predicted_goals_nn[0]:.2f}")

print(f"Naive: Przewidywana liczba goli w meczu: {naive_mse:.2f} \n")

print("-------------------------")
print("SUMMARY")

predicted_goals = [predicted_goals_linear[0], predicted_goals_polynomial[0], predicted_goals_decisionTree[0], predicted_goals_rf[0], predicted_goals_gb[0], 
                   predicted_goals_xgb[0], predicted_goals_nn[0], naive_mse]

valid_predicted = [value for value in predicted_goals if 0.01 < value <= 6.00]
if valid_predicted:
    summary_prediction = sum(valid_predicted) / len(valid_predicted)
else:
    summary_prediction = 0.0

print(f"Średnia przewidywana liczba goli: {summary_prediction:.2f}")
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