from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np


def train_and_visualize_rf_with_hyperparam_tuning(df, flux_column, X_columns):
    """
    Train a Random Forest Regressor on a given DataFrame, perform hyperparameter tuning, and visualize feature importance.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        date_column (str): Name of the column containing the date.
        X_columns (list): List of column names representing the features.

    Returns:
        float: Best Mean R-squared (R²) score after hyperparameter tuning.
        float: Best Mean Absolute Error (MAE) after hyperparameter tuning.
        float: Best Mean Root Mean Squared Error (RMSE) after hyperparameter tuning.
        pd.DataFrame: DataFrame containing feature importance scores of the best model.
        dict: Dictionary containing the best hyperparameters.

    """
    # Extract features and target variable
    X = df[X_columns]
    y = df[flux_column]

    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X, y)

    # Get the best hyperparameters from the search
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Get the best model from the search
    best_rf = grid_search.best_estimator_

    # Cross-validation with the best model
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    scores = cross_validate(best_rf, X, y, cv=cv, scoring=scoring, return_train_score=True)

    # Compute metrics using the best model
    best_r2 = np.mean(scores['test_r2'])
    best_mae = -np.mean(scores['test_neg_mean_absolute_error'])
    best_rmse = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))

    print("Best R²:", best_r2)
    print("Best MAE:", best_mae)
    print("Best RMSE:", best_rmse)

    # Feature importances of the best model
    importances = best_rf.feature_importances_

    # Create a DataFrame for feature importances
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Extract the top 5 features
    top_5_features = importance_df.nlargest(5, 'Importance')

    # Plot feature importances
    importance_df = importance_df.sort_values('Importance', ascending=False)
    # ax = importance_df.plot(kind='barh', x='Feature', y='Importance', figsize=(10, 6))

    # ax.set_title('Feature Importance')
    # ax.set_xlabel('Importance')
    # ax.set_ylabel('Feature')

    return best_r2, best_mae, best_rmse, importance_df, best_params, top_5_features