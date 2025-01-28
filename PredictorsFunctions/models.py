import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, r2_score

from PredictorsFunctions.modelsUtilities import set_model_details, calculate_cross_val_score, print_model_info

def naive_model(X_train, y_train, y_test):
    naive_pred = np.median(y_train)
    naive_mse = np.mean((y_test - naive_pred) ** 2)

    set_model_details("Naive", naive_mse, 0)
    calculate_cross_val_score("Naive", DummyRegressor(strategy='median'), True, X_train, y_train)
    print(f"\nNaive Baseline MSE: {round(naive_mse, 2)}")

    return naive_mse

def linear_model(X_train, y_train, X_test, y_test):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)

    set_model_details("Linear", mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
    calculate_cross_val_score("Linear", lin_reg, False, X_train, y_train)
    print_model_info("Regresja Liniowa:", "Linear")

    return lin_reg

def polynomial_model(X_train, y_train, X_test, y_test):
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    ridge_reg = Ridge(alpha=1.0)
    ridge_reg.fit(X_train_poly, y_train)
    
    y_pred_poly = ridge_reg.predict(X_test_poly)

    set_model_details("Polynomial", mean_squared_error(y_test, y_pred_poly), r2_score(y_test, y_pred_poly))    
    calculate_cross_val_score("Polynomial", ridge_reg, False, X_train, y_train, poly, "poly")
    print_model_info("Regresja Wielomianowa (stopień 2):", "Polynomial")
    return poly, ridge_reg    
    
def decisionTree_model(X_train, y_train, X_test, y_test):
    tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree_reg.fit(X_train, y_train)

    y_pred_tree = tree_reg.predict(X_test)
    y_pred_tree = np.maximum(0, y_pred_tree)

    set_model_details("DecisionTree", mean_squared_error(y_test, y_pred_tree), r2_score(y_test, y_pred_tree))
    calculate_cross_val_score("DecisionTree", tree_reg, False, X_train, y_train)
    print_model_info("Drzewo decyzyjne:", "DecisionTree")

    return tree_reg

def randomForest_model(X_train, y_train, X_test, y_test):
    rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_regressor.fit(X_train, y_train)

    y_pred_rf = rf_regressor.predict(X_test)

    set_model_details("RandomForest", mean_squared_error(y_test, y_pred_rf), r2_score(y_test, y_pred_rf))
    calculate_cross_val_score("RandomForest", rf_regressor, False, X_train, y_train)
    print_model_info("Random Forest:", "RandomForest")

    return rf_regressor

def gradientBoosting_model(X_train, y_train, X_test, y_test):
    gb_regressor = GradientBoostingRegressor(random_state=42)
    gb_regressor.fit(X_train, y_train)

    y_pred_gb = gb_regressor.predict(X_test)

    set_model_details("GradientBoosting", mean_squared_error(y_test, y_pred_gb), r2_score(y_test, y_pred_gb))
    calculate_cross_val_score("GradientBoosting", gb_regressor, False, X_train, y_train)
    print_model_info("Gradient Boosting:", "GradientBoosting")

    return gb_regressor

def xgbRegressor_model(X_train, y_train, X_test, y_test):
    xgb_reg = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb_reg.fit(X_train, y_train)

    y_pred_xgb = xgb_reg.predict(X_test)

    set_model_details("XGBoost", mean_squared_error(y_test, y_pred_xgb), r2_score(y_test, y_pred_xgb))
    calculate_cross_val_score("XGBoost", xgb_reg, False, X_train, y_train)
    print_model_info("XGBoost:", "XGBoost")

    return xgb_reg

def create_nn_model(X_train):    
    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='relu')
    ])

    nn_model.compile(optimizer='adam', loss='mse')
    return nn_model

def neuralNetwork_model(X_train, y_train, X_test, y_test):
    set_model_details("NeuralNetwork", 0, 0)
    calculate_cross_val_score("NeuralNetwork", KerasRegressor(model=lambda: create_nn_model(X_train), epochs=50, batch_size=4, verbose=0), False, X_train, y_train)
    nn_model = create_nn_model(X_train)
    nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=4, verbose=0)
    y_pred_nn = nn_model.predict(X_test).flatten()

    set_model_details("NeuralNetwork", mean_squared_error(y_test, y_pred_nn), r2_score(y_test, y_pred_nn))
    print_model_info("Neural network:", "NeuralNetwork")

    return nn_model