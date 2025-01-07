import xgboost as xgb
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_validate
import numpy as np
import pandas as pd

def train_and_visualize_xgb_with_hyperparam_tuning(X_train, y_train, X_columns):
    """
    Train an XGBoost Regressor on given training data, perform hyperparameter tuning, and visualize feature importance.

    Parameters:
        X_train (pd.DataFrame): DataFrame containing the training features.
        y_train (pd.Series): Series containing the training target variable.
        X_columns (list): List of column names representing the features.

    Returns:
        float: Best Mean R-squared (R²) score after hyperparameter tuning.
        float: Best Mean Absolute Error (MAE) after hyperparameter tuning.
        float: Best Mean Root Mean Squared Error (RMSE) after hyperparameter tuning.
        pd.DataFrame: DataFrame containing feature importance scores of the best model.
        dict: Dictionary containing the best hyperparameters.
        xgb.XGBRegressor: The best fitted XGBoost model.
    """
    # Initialize XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'colsample_bytree': [0.3, 0.7, 1.0],
        'subsample': [0.6, 0.8, 1.0]
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters from the search
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Get the best model from the search
    best_xgb = grid_search.best_estimator_

    # Cross-validation with the best model
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    scores = cross_validate(best_xgb, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

    # Compute metrics using the best model
    best_r2 = np.mean(scores['test_r2'])
    best_mae = -np.mean(scores['test_neg_mean_absolute_error'])
    best_rmse = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))

    print("Best R²:", best_r2)
    print("Best MAE:", best_mae)
    print("Best RMSE:", best_rmse)

    # Feature importances of the best model
    importances = best_xgb.feature_importances_

    # Create a DataFrame for feature importances
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Extract the top 5 features
    top_5_features = importance_df.nlargest(5, 'Importance')

    # Optionally, plot feature importances
    # importance_df = importance_df.sort_values('Importance', ascending=False)
    # ax = importance_df.plot(kind='barh', x='Feature', y='Importance', figsize=(10, 6))
    # ax.set_title('Feature Importance')
    # ax.set_xlabel('Importance')
    # ax.set_ylabel('Feature')

    return best_r2, best_mae, best_rmse, importance_df, best_params, best_xgb
