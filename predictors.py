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
from termcolor import colored

model_details = {}

def calculate_cross_score(model, name):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_mse = -scores.mean()
    print(f"Cross-validated MSE for {name}: {mean_mse:.2f}")

def set_model_details(modelName, y_test, y_pred):
    model_details[modelName] = {
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "Prediction": 0
    }

def print_model_info(header, modelName):
    print(header)
    print(f"Mean Squared Error: {model_details[modelName]['MSE']:.2f}")
    print(f"R2 Score: {model_details[modelName]['R2']:.2f} \n")    

def set_and_print_model_prediction(header, modelName, value):
    model_details[modelName]["Prediction"] = value
    print(colored(f"{header}: Przewidywana liczba goli w meczu: {value:.2f}", "green"))

#region DataInitialization
data_json = load_from_file()

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

set_model_details("Linear", y_test, y_pred)
print_model_info("\nRegresja Liniowa:", "Linear")
#endregion

#region PolynomialRegression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

y_pred_poly = poly_reg.predict(X_test_poly)
y_pred_poly = np.maximum(0, y_pred_poly)

set_model_details("Polynomial", y_test, y_pred_poly)
print_model_info("Regresja Wielomianowa (stopień 2):", "Polynomial")
#endregion

#region DecisionTree
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
# calculate_cross_score(tree_reg, "DecisionTree")
tree_reg.fit(X_train, y_train)

y_pred_tree = tree_reg.predict(X_test)
y_pred_tree = np.maximum(0, y_pred_tree)

set_model_details("DecisionTree", y_test, y_pred_tree)
print_model_info("Drzewo decyzyjne:", "DecisionTree")
#endregion

#region RandomForest
rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
# calculate_cross_score(rf_regressor, "RandomForest")
rf_regressor.fit(X_train, y_train)

y_pred_rf = rf_regressor.predict(X_test)

set_model_details("RandomForest", y_test, y_pred_rf)
print_model_info("Random Forest:", "RandomForest")
#endregion

#region GradientBoosting
gb_regressor = GradientBoostingRegressor(random_state=42)
# calculate_cross_score(gb_regressor, "Gradient")
gb_regressor.fit(X_train, y_train)

y_pred_gb = gb_regressor.predict(X_test)

set_model_details("GradientBoosting", y_test, y_pred_gb)
print_model_info("Gradient Boosting:", "GradientBoosting")
#endregion

#region XGBRegressor
xgb_reg = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
# calculate_cross_score(xgb_reg, "XGBRegressor")
xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
# y_pred_xgb = np.maximum(0, y_pred_xgb)

set_model_details("XGBoost", y_test, y_pred_xgb)
print_model_info("XGBoost:", "XGBoost")
#endregion

#region NeuralNetwork
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='relu')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=4, verbose=0)

y_pred_nn = model.predict(X_test).flatten()

set_model_details("NeuralNetwork", y_test, y_pred_nn)
print_model_info("Neural network:", "NeuralNetwork")
#endregion

#region Prediction
X_predict = predict_df[['home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg',
                        'hTeam_goals_all_avg', 'aTeam_goals_all_avg', 'hTeam_conceded_all_avg', 'aTeam_conceded_all_avg',
                        'hTeam_rivals_rating_avg', 'aTeam_rivals_rating_avg',  'h2h_hTeam_goals_all_avg', 'h2h_hTeam_conceded_all_avg',
                        'h2h_hTeam_home_goals_avg', 'h2h_hTeam_home_conceded_avg', 'h2h_aTeam_goals_all_avg', 'h2h_aTeam_conceded_all_avg',
                        'h2h_aTeam_away_goals_avg', 'h2h_aTeam_away_conceded_avg']]

set_and_print_model_prediction("Regresja liniowa", "Linear", lin_reg.predict(X_predict)[0])
set_and_print_model_prediction("Regresja wielomianowa", "Polynomial", poly_reg.predict(poly.transform(X_predict))[0])
set_and_print_model_prediction("Drzewo decyzyjne", "DecisionTree", tree_reg.predict(X_predict)[0])
set_and_print_model_prediction("Random forest", "RandomForest", rf_regressor.predict(X_predict)[0])
set_and_print_model_prediction("Gradient Boosting", "GradientBoosting", gb_regressor.predict(X_predict)[0])
set_and_print_model_prediction("XGBoost", "XGBoost", xgb_reg.predict(X_predict)[0])
set_and_print_model_prediction("Neural", "NeuralNetwork", model.predict(X_predict, verbose=0).flatten()[0])
print(colored(f"Naive: Przewidywana liczba goli w meczu: {naive_mse:.2f}", "green"))

print("-------------------------")
print("SUMMARY")

predicted_goals = [model_details["Linear"]["Prediction"], model_details["Polynomial"]["Prediction"], model_details["DecisionTree"]["Prediction"], 
                   model_details["RandomForest"]["Prediction"], model_details["GradientBoosting"]["Prediction"], 
                   model_details["XGBoost"]["Prediction"], model_details["NeuralNetwork"]["Prediction"], naive_mse]

valid_predicted = [value for value in predicted_goals if 0.01 < value <= 6.00]
if valid_predicted:
    summary_prediction = sum(valid_predicted) / len(valid_predicted)
else:
    summary_prediction = 0.0

print(colored(f"Średnia przewidywana liczba goli: {summary_prediction:.2f}", "green"))
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