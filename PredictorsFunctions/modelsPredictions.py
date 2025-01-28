from PredictorsFunctions.modelsUtilities import set_and_print_model_prediction
from termcolor import colored
from tabulate import tabulate

def prepare_X_predict(predict_df):
    X_predict = predict_df[['home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg',
                        'hTeam_goals_all_avg', 'aTeam_goals_all_avg', 'hTeam_conceded_all_avg', 'aTeam_conceded_all_avg',
                        'hTeam_rivals_rating_avg', 'aTeam_rivals_rating_avg',  'h2h_hTeam_goals_all_avg', 'h2h_hTeam_conceded_all_avg',
                        'h2h_hTeam_home_goals_avg', 'h2h_hTeam_home_conceded_avg', 'h2h_aTeam_goals_all_avg', 'h2h_aTeam_conceded_all_avg',
                        'h2h_aTeam_away_goals_avg', 'h2h_aTeam_away_conceded_avg']]
    
    return X_predict

def predict_goals_naive(naive_mse):
    set_and_print_model_prediction("Naive", "", naive_mse)    

def predict_goals_linear(X_predict, lin_reg):    
    set_and_print_model_prediction("Regresja liniowa", "Linear", lin_reg.predict(X_predict)[0])

def predict_goals_polynomial(X_predict, poly, poly_reg):
    set_and_print_model_prediction("Regresja wielomianowa", "Polynomial", poly_reg.predict(poly.transform(X_predict))[0])

def predict_goals_decisionTree(X_predict, tree_reg):
    set_and_print_model_prediction("Drzewo decyzyjne", "DecisionTree", tree_reg.predict(X_predict)[0])

def predict_goals_randomForest(X_predict, rf_regressor):
    set_and_print_model_prediction("Random forest", "RandomForest", rf_regressor.predict(X_predict)[0])

def predict_goals_gradientBoosting(X_predict, gb_regressor):
    set_and_print_model_prediction("Gradient Boosting", "GradientBoosting", gb_regressor.predict(X_predict)[0])

def predict_goals_xgBoost(X_predict, xgb_reg):
    set_and_print_model_prediction("XGBoost", "XGBoost", xgb_reg.predict(X_predict)[0])

def predict_goals_neuralNetwork(X_predict, nn_model):
    set_and_print_model_prediction("Neural", "NeuralNetwork", nn_model.predict(X_predict, verbose=0).flatten()[0])

def prediction_summary(models_details):    
    predicted_goals = [
        models_details[model]["Prediction"]
        for model in ["Linear", "Polynomial", "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "NeuralNetwork"]
        if model in models_details
    ]

    print(colored(f"\nŚrednia przewidywana liczba goli: {calculate_mean_prediction_goals(predicted_goals):.2f} \n", "green"))

def calculate_mean_prediction_goals(predicted_goals):
    valid_predicted = [value for value in predicted_goals if 0.01 < value <= 6.00]
    if valid_predicted:
        summary_prediction = sum(valid_predicted) / len(valid_predicted)
    else:
        summary_prediction = 0.0

    return summary_prediction

def summary_table(model_details):
    table_data = []
    for model_name, details in model_details.items():
        table_data.append([
            model_name,
            model_details[model_name]["MSE"],
            model_details[model_name]["R2"],
            model_details[model_name]["CrossValScore"],
            model_details[model_name]["Prediction"],
        ])

    headers = ["Model", "MSE (Lower Better)", "R2 Score (Higher Better - max value: 1)", "Cross Validation Score (Lower Better)", "Predicted goals"]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))