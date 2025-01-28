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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

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
    param_grid = {
        'max_depth': [2, 3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10]
    }

    tree_reg = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_tree = grid_search.best_estimator_
    
    y_pred_tree = np.maximum(0, best_tree.predict(X_test))

    set_model_details("DecisionTree", mean_squared_error(y_test, y_pred_tree), r2_score(y_test, y_pred_tree))    
    calculate_cross_val_score("DecisionTree", best_tree, False, X_train, y_train)
    print_model_info("Drzewo decyzyjne:", "DecisionTree")
    
    return best_tree

def randomForest_model(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    rand_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error')
    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_        

    y_pred_rf = np.maximum(0, best_rf.predict(X_test))

    set_model_details("RandomForest", mean_squared_error(y_test, y_pred_rf), r2_score(y_test, y_pred_rf))
    calculate_cross_val_score("RandomForest", best_rf, False, X_train, y_train)
    print_model_info("Random Forest:", "RandomForest")

    return best_rf

def gradientBoosting_model(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }   

    grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_gb = grid_search.best_estimator_

    y_pred_gb = np.maximum(0, best_gb.predict(X_test))

    set_model_details("GradientBoosting", mean_squared_error(y_test, y_pred_gb), r2_score(y_test, y_pred_gb))
    calculate_cross_val_score("GradientBoosting", best_gb, False, X_train, y_train)
    print_model_info("Gradient Boosting:", "GradientBoosting")

    return best_gb

def xgbRegressor_model(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_    

    y_pred_xgb = best_xgb.predict(X_test)

    set_model_details("XGBoost", mean_squared_error(y_test, y_pred_xgb), r2_score(y_test, y_pred_xgb))
    calculate_cross_val_score("XGBoost", best_xgb, False, X_train, y_train)
    print_model_info("XGBoost:", "XGBoost")

    return best_xgb

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