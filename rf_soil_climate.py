import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.cm as cm


def train_and_visualize_rf_with_hyperparam_tuning(X_train, y_train, X_columns):
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    importances = best_rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_columns,
        'Importance': importances
    })

    return best_rf, feature_importance_df


def main():
    # Load data
    data = pd.read_csv('merged_data.csv')
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], errors='coerce')
    data['YEAR'] = data['TIMESTAMP'].dt.year

    # Rename feature columns
    renamed_columns = {
        'bdod_0-5cm_mean': 'Bd',
        'cec_0-5cm_mean': 'CEC',
        'phh2o_0-5cm_mean': 'pH',
        'soc_0-5cm_mean': 'SOC',
        'nitrogen_0-5cm_mean': 'TN',
        'P_F': 'Pressure',
        'TA_F': 'Air Temperature',
        'WTD_F': 'Water Table Depth',
        'NEE_F': 'NEEE'
    }

    # Soil-related features for color differentiation
    soil_features = ['Bd', 'CEC', 'pH', 'SOC', 'TN']

    # Map KÖPPEN to categories and filter out 'Arid'
    koppen_descriptions = {
        'Dfc': 'Continental', 'Cfb': 'Temperate', 'Csa': 'Temperate',
        'Cfa': 'Temperate', 'Dfb': 'Continental', 'Dfa': 'Continental',
        'Dwc': 'Continental', 'ET': 'Polar', 'Am': 'Tropical',
        'Cwa': 'Temperate', 'Aw': 'Tropical', 'Dfd': 'Continental',
        'Af': 'Tropical', 'Cwc': 'Temperate', 'Dwa': 'Continental',
        'Bsh': 'Arid'
    }
    data['CLIMATE'] = data['KOPPEN'].map(koppen_descriptions)
    data = data[data['CLIMATE'] != 'Arid']

    # Define columns
    columns_to_group = ['P_F', 'TA_F', 'WTD_F', 'NEE_F', 'FCH4_F']
    soil_columns = ['bdod_0-5cm_mean', 'cec_0-5cm_mean', 'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean']

    # Grouped datasets
    grouped_data = data.groupby(['SITE_ID', 'YEAR', 'CLIMATE'])[columns_to_group + soil_columns].mean().reset_index()
    grouped_data.dropna(inplace=True)

    # Prepare results for both cases
    results = {}

    for include_soil_data in [True, False]:
        cols = columns_to_group + (soil_columns if include_soil_data else [])
        X = grouped_data[cols].drop(columns='FCH4_F')
        y = grouped_data['FCH4_F']
        climates = grouped_data['CLIMATE']

        # Train-test split
        X_train, X_test, y_train, y_test, climates_train, climates_test = train_test_split(
            X, y, climates, test_size=0.3, random_state=42)

        # Train and get feature importance
        best_rf, feature_importance_df = train_and_visualize_rf_with_hyperparam_tuning(X_train, y_train, X.columns)

        # Rename features for plotting
        feature_importance_df['Feature'] = feature_importance_df['Feature'].map(renamed_columns).fillna(feature_importance_df['Feature'])
        feature_importance_df['Color'] = feature_importance_df['Feature'].apply(
            lambda x: '#FFA500' if x in soil_features else '#4682B4')  # Orange for soil, Steel blue for others

        # Predictions
        y_pred = best_rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results['with_soil' if include_soil_data else 'without_soil'] = {
            'feature_importance': feature_importance_df,
            'observed': y_test,
            'predicted': y_pred,
            'climates': climates_test,
            'r2': r2,
            'rmse': rmse
        }

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

    for i, (key, result) in enumerate(results.items()):
        # Feature Importance Plot
        importance_df = result['feature_importance'].sort_values('Importance', ascending=False)
        axs[i, 0].barh(importance_df['Feature'], importance_df['Importance'], color=importance_df['Color'])
        axs[i, 0].set_title(f"Feature Importance ({key.replace('_', ' ')})", fontsize=16)
        axs[i, 0].set_xlabel('Importance', fontsize=16)
        axs[i, 0].set_ylabel('Feature', fontsize=16)

        # Observed vs Predicted Plot with colors by climate
        climates = result['climates']
        unique_climates = climates.unique()
        color_map = {climate: cm.tab10(i / len(unique_climates)) for i, climate in enumerate(unique_climates)}

        for climate in unique_climates:
            idx = climates == climate
            axs[i, 1].scatter(result['observed'][idx], result['predicted'][idx],
                              color=color_map[climate], alpha=0.7, label=climate)

        axs[i, 1].plot([result['observed'].min(), result['observed'].max()],
                       [result['observed'].min(), result['observed'].max()], 'r--', linewidth=2)
        axs[i, 1].set_title(f"Observed vs Predicted ({key.replace('_', ' ')})", fontsize=16)
        axs[i, 1].set_xlabel('Observed Mean Annual Flux (nmol CH₄ m⁻² s⁻¹)', fontsize=16)
        axs[i, 1].set_ylabel('Predicted Mean Annual Flux (nmol CH₄ m⁻² s⁻¹)', fontsize=16)
        axs[i, 1].text(0.95, 0.05, f"R² = {result['r2']:.2f}\nRMSE = {result['rmse']:.2f}",
                       transform=axs[i, 1].transAxes, fontsize=14, va='bottom', ha='right')
        axs[i, 1].legend(title="Climate", fontsize=14, title_fontsize=16, loc="best")

    # Save the figure as a high-resolution PNG
    plt.savefig('random_forest_results_highres.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
