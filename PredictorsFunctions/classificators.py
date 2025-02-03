from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

def xgb_classificator(X_train, y_train, X_test, y_test):    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    y_train_class = (y_train > 2.5).astype(int)
    y_test_class = (y_test > 2.5).astype(int)

    xgb_clf = XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train_class)

    best_clf = grid_search.best_estimator_
    y_pred_proba = best_clf.predict_proba(X_test)[:, 1]        

    threshold = 0.5
    y_pred_class = (y_pred_proba > threshold).astype(int) # 1 - over, 0 - under
    
    accuracy = accuracy_score(y_test_class, best_clf.predict(X_test))
    roc_auc = roc_auc_score(y_test_class, y_pred_proba)
    print(f"Accuracy (how good model predict result): {accuracy:.4f}, ROC AUC (great value = 1; value < 0.5 - worse than random): {roc_auc:.4f}")
    print(f"Classificator summary: {y_pred_class}")