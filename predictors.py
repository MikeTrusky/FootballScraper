from PredictorsFunctions.dataInitializer import prepare_dataFrames
from PredictorsFunctions.dataInitializer import prepare_sets

import PredictorsFunctions.models as models
import PredictorsFunctions.modelsPredictions as predictors
from PredictorsFunctions.modelsUtilities import models_details

from sklearn.preprocessing import StandardScaler

def main():
    train_df, predict_df = prepare_dataFrames()
    X_train, X_test, y_train, y_test = prepare_sets(train_df)

    scaler = StandardScaler()

    naive_mse       = models.naive_model(X_train, y_train, y_test)
    lin_reg         = models.linear_model(scaler.fit_transform(X_train), y_train, scaler.transform(X_test), y_test)
    poly, poly_reg  = models.polynomial_model(scaler.fit_transform(X_train), y_train, scaler.transform(X_test), y_test)
    tree_reg        = models.decisionTree_model(X_train, y_train, X_test, y_test)
    rf_regressor    = models.randomForest_model(X_train, y_train, X_test, y_test)
    gb_regressor    = models.gradientBoosting_model(X_train, y_train, X_test, y_test)
    xgb_reg         = models.xgbRegressor_model(X_train, y_train, X_test, y_test)
    nn_model        = models.neuralNetwork_model(X_train, y_train, X_test, y_test)

    X_predict = predictors.prepare_X_predict(predict_df)
    predictors.predict_goals_naive(naive_mse)
    predictors.predict_goals_linear(scaler.transform(X_predict), lin_reg)
    predictors.predict_goals_polynomial(scaler.transform(X_predict), poly, poly_reg)
    predictors.predict_goals_decisionTree(X_predict, tree_reg)
    predictors.predict_goals_randomForest(X_predict, rf_regressor)
    predictors.predict_goals_gradientBoosting(X_predict, gb_regressor)
    predictors.predict_goals_xgBoost(X_predict, xgb_reg)
    predictors.predict_goals_neuralNetwork(X_predict, nn_model)

    predictors.summary_table(models_details)
    predictors.prediction_summary(models_details)

main()