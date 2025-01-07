import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_validate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def train_and_visualize_rf_with_hyperparam_tuning(X_train, y_train, X_columns, include_soil_data=True):
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    best_rf = grid_search.best_estimator_

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    scores = cross_validate(best_rf, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

    best_r2 = np.mean(scores['test_r2'])
    best_mae = -np.mean(scores['test_neg_mean_absolute_error'])
    best_rmse = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))

    print("Best R²:", best_r2)
    print("Best MAE:", best_mae)
    print("Best RMSE:", best_rmse)

    importances = best_rf.feature_importances_
    feature_names = X_columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    top_5_features = importance_df.nlargest(5, 'Importance')

    return best_r2, best_mae, best_rmse, importance_df, best_params, top_5_features, best_rf

def main(include_soil_data=True):
    data = pd.read_csv('merged_data.csv')
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], errors='coerce')
    data['YEAR'] = data['TIMESTAMP'].dt.year

    # Climate boxplot
    # Köppen Description Dictionary
    koppen_descriptions = {
        'Dfc': 'Continental',
        'Cfb': 'Temperate',
        'Csa': 'Temperate',
        'Cfa': 'Temperate',
        'Dfb': 'Continental',
        'Dfa': 'Continental',
        'Dwc': 'Continental',
        'ET': 'Polar',
        'Am': 'Tropical',
        'Cwa': 'Temperate',
        'Aw': 'Tropical',
        'Dfd': 'Continental',
        'Af': 'Tropical',
        'Cwc': 'Temperate',
        'Dwa': 'Continental',
        'Bsh': 'Arid'
    }

    # Map the 'KOPPEN' column to climate categories using the koppen_descriptions dictionary
    data['CLIMATE'] = data['KOPPEN'].map(koppen_descriptions)

    # Remove Arid Climates
    data = data[data['CLIMATE'] != 'Arid']

    columns_to_group = ['P_F', 'TA_F', 'WTD_F', 'NEE_F', 'FCH4_F']
    soil_columns = ['bdod_0-5cm_mean', 'cec_0-5cm_mean', 'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean']
    if include_soil_data:
        columns_to_group.extend(soil_columns)

    grouped_data = data.groupby(['SITE_ID', 'YEAR'])[columns_to_group].mean().reset_index()
    grouped_data.dropna(inplace=True)

    X_columns = [col for col in columns_to_group if col != 'FCH4_F']
    X = grouped_data[X_columns]
    y = grouped_data['FCH4_F']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    best_r2, best_mae, best_rmse, importance_df, best_params, top_5_features, best_rf = train_and_visualize_rf_with_hyperparam_tuning(X_train, y_train, X.columns, include_soil_data)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()

    y_pred = best_rf.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('Observed CH4 Flux')
    plt.ylabel('Predicted CH4 Flux')
    plt.title(f'CH4 Flux Prediction (With Soil Data: {include_soil_data})')

    # Add R² value to the plot
    plt.text(0.9, 0.1, f'R² = {test_r2:.2f}', ha='right', va='center', transform=plt.gca().transAxes)

    # Add 1:1 line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.show()

    return {
        'With_soil_data': include_soil_data,
        'RMSE': test_rmse,
        'R2_Score': test_r2
    }

if __name__ == "__main__":
    results = []

    result_with_soil = main(include_soil_data=True)
    results.append(result_with_soil)

    result_without_soil = main(include_soil_data=False)
    results.append(result_without_soil)

    results_df = pd.DataFrame(results)
    print(results_df)

