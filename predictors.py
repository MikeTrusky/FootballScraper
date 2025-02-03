import PredictorsFunctions.dataInitializer as initializer

import PredictorsFunctions.models as models
import PredictorsFunctions.modelsPredictions as predictors
from PredictorsFunctions.modelsUtilities import models_details
import PredictorsFunctions.predictionSummary as summary
import PredictorsFunctions.classificators as classificators

from sklearn.preprocessing import StandardScaler

def main(naive, linear, polynomial, decisionTree, randomForest, gradientBoosting, xgbRegressor, neural, poisson, showSummary, xgbClassificator):
    train_df, predict_df = initializer.prepare_dataFrames()    
    train_df, predict_df = initializer.remove_redundant_features(train_df, predict_df)
    available_features = initializer.get_available_features(train_df)
    X_train, X_test, y_train, y_test, X_predict = initializer.prepare_sets(train_df, predict_df, available_features)
    assert set(train_df.columns) - {'total_goals'} == set(predict_df.columns) - {'total_goals'}, "Kolumny nie są zgodne!"
    assert set(train_df.columns) - {'total_goals'} == set(X_predict.columns) - {'total_goals'}, "Kolumny nie są zgodne!"

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_predict_scaled = scaler.transform(X_predict)
    
    if naive:
        naive_mse           = models.naive_model(X_train, y_train, y_test)
        predictors.predict_goals_naive(naive_mse)

    if linear:        
        lin_reg             = models.linear_model(X_train_scaled, y_train, X_test_scaled, y_test)    
        predictors.predict_goals_linear(X_predict_scaled, lin_reg)

    if polynomial:
        poly_reg_pipeline   = models.polynomial_model(X_train_scaled, y_train, X_test_scaled, y_test)
        predictors.predict_goals_polynomial(X_predict_scaled, poly_reg_pipeline)
    
    if decisionTree:
        tree_reg            = models.decisionTree_model(X_train, y_train, X_test, y_test)
        predictors.predict_goals_decisionTree(X_predict, tree_reg)

    if randomForest:
        rf_regressor        = models.randomForest_model(X_train, y_train, X_test, y_test)
        predictors.predict_goals_randomForest(X_predict, rf_regressor)

    if gradientBoosting:
        gb_regressor        = models.gradientBoosting_model(X_train, y_train, X_test, y_test)
        predictors.predict_goals_gradientBoosting(X_predict, gb_regressor)

    if xgbRegressor:
        xgb_reg             = models.xgbRegressor_model(X_train, y_train, X_test, y_test) # the longest time to learn
        predictors.predict_goals_xgBoost(X_predict, xgb_reg)

    if neural:
        nn_model            = models.neuralNetwork_model(X_train, y_train, X_test, y_test)
        predictors.predict_goals_neuralNetwork(X_predict, nn_model)

    if poisson:
        poisson_model       = models.poisson_model(X_train, y_train, X_test, y_test)    
        predictors.predict_goals_poisson(X_predict, poisson_model)

    if showSummary:
        summary.summary_table(models_details)
        summary.prediction_summary(models_details)

    if xgbClassificator:
        classificators.xgb_classificator(X_train, y_train, X_test, y_test)

main(
    naive=True,
    linear=True,
    polynomial=True,
    decisionTree=True,
    randomForest=True,
    gradientBoosting=True,
    xgbRegressor=True,
    neural=True,
    poisson=True,
    showSummary=True,
    xgbClassificator=True
)