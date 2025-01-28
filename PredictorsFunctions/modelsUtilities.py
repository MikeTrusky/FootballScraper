from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from termcolor import colored

models_details = {}

def calculate_cross_val_score(modelName, crossValModel, useStandarScaler, X_train, y_train, additionalStep = None, additionalStepName = ""):
    pipelineSteps = []
    if additionalStep != None:
        pipelineSteps.append((f'{additionalStepName}', additionalStep))
    if useStandarScaler:
        pipelineSteps.append(('scaler', StandardScaler()))
    pipelineSteps.append(('model', crossValModel))
    pipeline = Pipeline(pipelineSteps)
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error', error_score='raise')
    mean_mse = -scores.mean()
    models_details[modelName]["CrossValScore"] = round(mean_mse, 2)

def set_model_details(modelName, mseValue, r2Value):    
    models_details[modelName] = {
        "MSE": round(mseValue, 2),
        "R2": round(r2Value, 2),
        "CrossValScore": 0,
        "Prediction": 0
    }    

def update_model_details_CVS(modelName, crossValScoreValue):
    models_details[modelName]["CrossValScore"] = crossValScoreValue

def print_model_info(header, modelName):
    print(header + " done!")
    # print(f"Mean Squared Error: {models_details[modelName]['MSE']:.2f}")
    # print(f"Cross Validation Score: {models_details[modelName]['CrossValScore']:.2f}")
    # print(f"R2 Score: {models_details[modelName]['R2']:.2f} \n")

def set_and_print_model_prediction(header, modelName, value):
    if modelName != "":
        models_details[modelName]["Prediction"] = round(value, 2)
    # print(colored(f"{header}: Przewidywana liczba goli w meczu: {value:.2f}", "green"))