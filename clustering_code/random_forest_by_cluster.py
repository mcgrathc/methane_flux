from sklearn.cluster import KMeans
import pandas as pd
from annual_process_and_clustering_function import load_and_preprocess, standardize_data, perform_pca
from rf_function import train_and_visualize_rf_with_hyperparam_tuning
from random_forest_plotting_function import plot_feature_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def process_and_visualize_data(agg_function):
    """
    Processes data, performs clustering, trains random forest models,
    and saves the feature importance and model metrics.

    Parameters:
    agg_function (str): The aggregation function to be used ('sum', 'mean', 'min', 'max').
    """

    # Load and preprocess data
    df_grouped, num_cols = load_and_preprocess('merged_data.csv', agg_function=agg_function)
    df_grouped_scaled = standardize_data(df_grouped, num_cols)

    # Perform PCA
    pca_model, reduced_df, loadings = perform_pca(df_grouped_scaled)

    # Fit clustering model
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(reduced_df)

    # Add cluster labels
    df_grouped_scaled['cluster'] = clusters

    X_columns = ['P_F', 'TA_F', 'WTD_F', 'PA_F', 'G_F', 'H_F', 'NEE_F']
               # 'bdod_0-5cm_mean', 'cec_0-5cm_mean',
                # 'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean']

    # Initialize lists to store global min and max values for the diagonal line
    all_y_test = []
    all_predictions = []

    # Initialize list to store importance dataframes
    metrics_data = {'Cluster': [], 'R2': [], 'MAE': [], 'RMSE': []}
    importance_dfs = []

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Dictionary to map aggregation function to a more readable format
    agg_function_title_map = {
        'sum': 'Annual Sum of CH$_4$ Flux',
        'mean': 'Annual Mean of CH$_4$ Flux',
        'min': 'Annual Minimum of CH$_4$ Flux',
        'max': 'Annual Maximum of CH$_4$ Flux',
    }
    title = agg_function_title_map.get(agg_function, 'Annual CH$_4$ Flux')

    # Iterate over each cluster to plot their data points and collect importance data
    for cluster in [0, 1]:
        df_cluster = df_grouped_scaled[df_grouped_scaled['cluster'] == cluster]
        X = df_cluster[X_columns]
        y = df_cluster['FCH4_F']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model and get predictions
        r2, mae, rmse, imp_df, best_params, top_feat, best_rf = train_and_visualize_rf_with_hyperparam_tuning(
            X_train, y_train, X_columns
        )

        metrics_data['Cluster'].append(f'Cluster {cluster}')
        metrics_data['R2'].append(r2)
        metrics_data['MAE'].append(mae)
        metrics_data['RMSE'].append(rmse)

        predictions = best_rf.predict(X_test)

        # Store predictions for the diagonal line calculation
        all_y_test.extend(y_test)
        all_predictions.extend(predictions)

        # Plot scatterplot for observed vs predicted FCH4 values with different colors for each cluster
        plt.scatter(y_test, predictions, s=150, alpha=0.6, label=f'Cluster {cluster}')

        # Append the feature importance dataframe to the list
        if imp_df is not None and not imp_df.empty:
            importance_dfs.append(imp_df)
        else:
            print(f"Warning: Feature importance DataFrame for cluster {cluster} is empty or not returned.")

    plt.title(title)
    plt.xlabel('Observed CH$_4$ Flux')
    plt.ylabel('Predicted CH$_4$ Flux')

    # Plot the diagonal line using global min and max values from all clusters
    combined_min = min(min(all_y_test), min(all_predictions))
    combined_max = max(max(all_y_test), max(all_predictions))
    plt.plot([combined_min, combined_max], [combined_min, combined_max], 'k--', lw=2)

    plt.grid(True)
    plt.legend()
    plt.savefig(f'manuscript\\fch4_scatterplot_{agg_function}_no_soil.png')
    plt.show()

    # Check if importance_dfs is not empty before concatenating
    if not importance_dfs:
        raise ValueError("No feature importance DataFrames to concatenate.")

    # Merge and compare importance dfs
    importance_df = pd.concat(importance_dfs, axis=1, keys=['Cluster 0', 'Cluster 1'])
    importance_df.columns = importance_df.columns.map('_'.join)

    # Plot and save feature importance
    feature_importance_filename = f'manuscript\\feature_importance_rf_comparison_{agg_function}_no_soil.png'
    plot_feature_importance(importance_df, agg_function, feature_importance_filename)

    # Create and save metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_filename = f'manuscript\\cluster_rf_model_metrics_{agg_function}_no_soil.csv'
    metrics_df.to_csv(metrics_filename, index=False)


# Proces and visualize rf results for all agg.
process_and_visualize_data('sum')
process_and_visualize_data('mean')
process_and_visualize_data('min')
process_and_visualize_data('max')



from sklearn.cluster import KMeans
import pandas as pd
from annual_process_and_clustering_function import load_and_preprocess, standardize_data, perform_pca
from rf_function import train_and_visualize_rf_with_hyperparam_tuning
from random_forest_plotting_function import plot_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  # Import the r2_score function
import matplotlib.pyplot as plt
import seaborn as sns

def process_and_visualize_data(agg_function):
    # Load and preprocess data
    df_grouped, num_cols = load_and_preprocess('merged_data.csv', agg_function=agg_function)
    df_grouped_scaled = standardize_data(df_grouped, num_cols)

    # Perform PCA
    pca_model, reduced_df, loadings = perform_pca(df_grouped_scaled)

    # Fit clustering model
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(reduced_df)
    df_grouped_scaled['cluster'] = clusters

    X_columns = ['P_F', 'TA_F', 'WTD_F', 'PA_F', 'G_F', 'H_F', 'NEE_F',
                  'bdod_0-5cm_mean', 'cec_0-5cm_mean',
                'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean']

    # Initialize lists to store global min and max values for the diagonal line
    all_y_test = []
    all_predictions = []

    # Initialize list to store importance dataframes
    metrics_data = {'Cluster': [], 'R2': [], 'MAE': [], 'RMSE': []}
    importance_dfs = []

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    agg_function_title_map = {
        'sum': 'Annual Sum of CH$_4$ Flux',
        'mean': 'Annual Mean of CH$_4$ Flux',
        'min': 'Annual Minimum of CH$_4$ Flux',
        'max': 'Annual Maximum of CH$_4$ Flux',
    }
    title = agg_function_title_map.get(agg_function, 'Annual CH$_4$ Flux')

    for cluster in [0, 1]:
        df_cluster = df_grouped_scaled[df_grouped_scaled['cluster'] == cluster]
        X = df_cluster[X_columns]
        y = df_cluster['FCH4_F']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model and get predictions
        r2, mae, rmse, imp_df, best_params, top_feat, best_rf = train_and_visualize_rf_with_hyperparam_tuning(
            X_train, y_train, X_columns
        )

        predictions = best_rf.predict(X_test)
        r2_score_value = r2_score(y_test, predictions)  # Calculate R2 score

        metrics_data['Cluster'].append(f'Cluster {cluster}')
        metrics_data['R2'].append(r2_score_value)  # Update to use the calculated R2 score
        metrics_data['MAE'].append(mae)
        metrics_data['RMSE'].append(rmse)

        all_y_test.extend(y_test)
        all_predictions.extend(predictions)

        plt.scatter(y_test, predictions, s=150, alpha=0.6, label=f'Cluster {cluster}')
        plt.annotate(f'R² = {r2_score_value:.2f}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
                     backgroundcolor='white')

        if imp_df is not None and not imp_df.empty:
            importance_dfs.append(imp_df)

    plt.title(title)
    plt.xlabel('Observed CH$_4$ Flux')
    plt.ylabel('Predicted CH$_4$ Flux')

    combined_min = min(min(all_y_test), min(all_predictions))
    combined_max = max(max(all_y_test), max(all_predictions))
    plt.plot([combined_min, combined_max], [combined_min, combined_max], 'k--', lw=2)

    plt.grid(True)
    plt.legend()
    plt.savefig(f'manuscript\\fch4_scatterplot_{agg_function}.png')
    plt.show()

    if not importance_dfs:
        raise ValueError("No feature importance DataFrames to concatenate.")

    importance_df = pd.concat(importance_dfs, axis=1, keys=['Cluster 0', 'Cluster 1'])
    importance_df.columns = importance_df.columns.map('_'.join)

    feature_importance_filename = f'manuscript\\feature_importance_rf_comparison_{agg_function}.png'
    plot_feature_importance(importance_df, agg_function, feature_importance_filename)

    metrics_df = pd.DataFrame(metrics_data)
    metrics_filename = f'manuscript\\cluster_rf_model_metrics_{agg_function}.csv'
    metrics_df.to_csv(metrics_filename, index=False)

# Process and visualize rf results for all aggregations
process_and_visualize_data('sum')
process_and_visualize_data('mean')
process_and_visualize_data('min')
process_and_visualize_data('max')
