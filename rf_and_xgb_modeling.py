from sklearn.cluster import KMeans
import pandas as pd
from annual_process_and_clustering_function import load_and_preprocess, standardize_data, perform_pca
from rf_function import train_and_visualize_rf_with_hyperparam_tuning
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

    # Create and save metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_filename = f'manuscript\\cluster_rf_model_metrics_{agg_function}.csv'
    metrics_df.to_csv(metrics_filename, index=False)


# Proces and visualize rf results for all agg.
process_and_visualize_data('sum')
process_and_visualize_data('mean')
process_and_visualize_data('min')
process_and_visualize_data('max')


