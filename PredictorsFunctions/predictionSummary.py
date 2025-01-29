from termcolor import colored
from tabulate import tabulate
import numpy as np

def prediction_summary(models_details):    
    predicted_goals = [
        models_details[model]["Prediction"]
        for model in ["Linear", "Polynomial", "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "NeuralNetwork"]
        if model in models_details
    ]

    mse_values = [
        models_details[model]["MSE"]
        for model in ["Linear", "Polynomial", "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "NeuralNetwork"]
        if model in models_details
    ]

    r2_values = [
        models_details[model]["R2"]
        for model in ["Linear", "Polynomial", "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "NeuralNetwork"]
        if model in models_details
    ]

    print(colored(f"\nŚrednia przewidywana liczba goli biorąc pod uwagę wszystkie modele: ", "green") + colored(f"{calculate_mean_prediction_goals(predicted_goals):.2f}", "red"))
    print(colored(f"Średnia przewidywana liczba goli według średniej ważonej, odrzucającej modele z R2 < 0. Wykorzystanie 1/mse: ", "green") + colored(f"{weighted_mean_prediction(predicted_goals, mse_values, r2_values):.2f}", "red"))
    print(colored(f"Średnia przewidywana liczba goli według średniej ważonej, odrzucającej modele z R2 < 0. Wykorzystanie normalizacji i log(1/mse): ", "green") + colored(f"{weighted_mean_prediction(predicted_goals, mse_values, r2_values, 'log', True):.2f}", "red"))
    print(colored(f"Średnia przewidywana liczba goli według średniej ważonej, odrzucającej modele z R2 < 0. Wykorzystanie normalizacji i 1 - mse: ", "green") + colored(f"{weighted_mean_prediction(predicted_goals, mse_values, r2_values, 'minus', True):.2f}", "red"))    

def calculate_mean_prediction_goals(predicted_goals):
    valid_predicted = [value for value in predicted_goals if 0.01 < value <= 6.00]
    if valid_predicted:
        summary_prediction = sum(valid_predicted) / len(valid_predicted)
    else:
        summary_prediction = 0.0

    return summary_prediction

def normalize_mse(mse_values):
    min_mse, max_mse = min(mse_values), max(mse_values)
    return [(mse - min_mse) / (max_mse - min_mse) * 0.99 + 0.01 for mse in mse_values]

def weighted_mean_prediction(predicted_goals, mse_values, r2_values, method = "invert", doNormalize = False):
    mse_for_calculation = normalize_mse(mse_values) if doNormalize else mse_values    
    weights = []
    for mse, r2 in zip(mse_for_calculation, r2_values):
        if r2 <= 0:            
            weights.append(0)
        else:
            if method == "invert":
                weight = (1 / mse if mse > 0 else 1)
            elif method == "log":
                weight = np.log(1 / mse)
            elif method == "minus":
                weight = 1 - mse
            weight *= r2
            weights.append(weight)
    
    valid_predictions = [prediction for i, prediction in enumerate(predicted_goals) if weights[i] > 0]
    valid_weights = [weight for weight in weights if weight > 0]    

    return sum(p * w for p, w in zip(valid_predictions, valid_weights)) / sum(valid_weights) if valid_weights else 0.0

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

    table_data.sort(key=lambda x: x[1])

    headers = ["Model", "MSE (Lower Better)", "R2 Score (Higher Better - max value: 1)", "Cross Validation Score (Lower Better)", "Predicted goals"]
    headers = [colored(header, 'cyan') for header in headers]

    mse_values = [x[1] for x in table_data]
    r2_values = [x[2] for x in table_data]
    cvs_values = [x[3] for x in table_data]

    def colorize_row(row):
        model_name = row[0]
        mse = row[1]
        r2 = row[2]
        cvs = row[3]
        
        if model_name == "Naive":
            row = [colored(value, 'blue') for value in row]
        
        if mse == min(mse_values):
            row[1] = colored(row[1], 'green')
        
        if r2 == max(r2_values):
            row[2] = colored(row[2], 'green')
        
        if cvs == min(cvs_values):
            row[3] = colored(row[3], 'green')

        return row
    
    table_data = [colorize_row(row) for row in table_data]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))