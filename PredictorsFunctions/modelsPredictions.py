from PredictorsFunctions.modelsUtilities import set_model_prediction

def predict_goals_naive(naive_mse):
    set_model_prediction("Naive", "", naive_mse)    

def predict_goals_linear(X_predict, lin_reg):    
    set_model_prediction("Regresja liniowa", "Linear", lin_reg.predict(X_predict)[0])

def predict_goals_polynomial(X_predict, poly_reg_pipeline):
    set_model_prediction("Regresja wielomianowa", "Polynomial", poly_reg_pipeline.predict(X_predict)[0])

def predict_goals_decisionTree(X_predict, tree_reg):
    set_model_prediction("Drzewo decyzyjne", "DecisionTree", tree_reg.predict(X_predict)[0])

def predict_goals_randomForest(X_predict, rf_regressor):
    set_model_prediction("Random forest", "RandomForest", rf_regressor.predict(X_predict)[0])

def predict_goals_gradientBoosting(X_predict, gb_regressor):
    set_model_prediction("Gradient Boosting", "GradientBoosting", gb_regressor.predict(X_predict)[0])

def predict_goals_xgBoost(X_predict, xgb_reg):
    set_model_prediction("XGBoost", "XGBoost", xgb_reg.predict(X_predict)[0])

def predict_goals_neuralNetwork(X_predict, nn_model):
    set_model_prediction("Neural", "NeuralNetwork", nn_model.predict(X_predict, verbose=0).flatten()[0])

def predict_goals_poisson(X_predict, poison_model):
    set_model_prediction("Poisson reggresor", "Poisson", poison_model.predict(X_predict)[0])

def predict_goals_light(X_predict, light_model):
    set_model_prediction("LightGBM", "LightGBM", light_model.predict(X_predict)[0])