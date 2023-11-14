from sklearn.cluster import KMeans
import pandas as pd
from annual_process_and_clustering_function import load_and_preprocess, standardize_data, perform_pca
from rf_function import train_and_visualize_rf_with_hyperparam_tuning
from random_forest_plotting_function import plot_feature_importance


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

    X_columns = ['P_F', 'TA_F', 'WTD_F', 'PA_F', 'G_F', 'H_F', 'NEE_F',
                 'bdod_0-5cm_mean', 'cec_0-5cm_mean',
                 'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean']

    # Process each cluster and collect metrics
    metrics_data = {'Cluster': [], 'R2': [], 'MAE': [], 'RMSE': []}
    importance_dfs = []

    for cluster in [0, 1]:
        df_cluster = df_grouped_scaled[df_grouped_scaled['cluster'] == cluster]
        r2, mae, rmse, imp_df, best_params, top_feat = train_and_visualize_rf_with_hyperparam_tuning(
            df_cluster, flux_column='FCH4_F', X_columns=X_columns
        )

        metrics_data['Cluster'].append(f'Cluster {cluster}')
        metrics_data['R2'].append(r2)
        metrics_data['MAE'].append(mae)
        metrics_data['RMSE'].append(rmse)

        importance_dfs.append(imp_df)

    # Merge and compare importance dfs
    importance_df = pd.concat(importance_dfs, axis=1, keys=['Cluster 0', 'Cluster 1'])
    importance_df.columns = importance_df.columns.map('_'.join)

    # Create and save metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_filename = f'cluster_model_metrics_{agg_function}.csv'
    metrics_df.to_csv(metrics_filename, index=False)

    # Plot and save feature importance
    feature_importance_filename = f'feature_importance_comparison_{agg_function}.png'
    plot_feature_importance(importance_df, feature_importance_filename)


# Proces and visualize rf results for all agg.
process_and_visualize_data('sum')
process_and_visualize_data('mean')
process_and_visualize_data('min')
process_and_visualize_data('max')
