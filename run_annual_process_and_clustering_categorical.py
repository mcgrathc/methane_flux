from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import pandas as pd
from annual_process_and_clustering_function_categorical import load_and_preprocess, standardize_data, perform_pca, perform_clustering, generate_biplot, plot_correlation_matrix


def process_and_cluster(agg_function='sum'):
    """
    Process and cluster data using various algorithms.

    This function performs a series of data processing steps including loading, preprocessing,
    standardizing, PCA transformation, and clustering. It applies different clustering algorithms
    and calculates the silhouette score for each. It also generates biplots for visualizing the
    clustering results. The aggregation function used in the preprocessing can be varied.

    Parameters:
    agg_function (str, optional): The aggregation function to use in data preprocessing.
                                  Options are 'sum', 'mean', 'min', 'max'. Defaults to 'sum'.

    Returns:
    None: This function does not return a value but prints results and saves plots.
    """
    # Load and pre-process data
    df_grouped, num_cols, = load_and_preprocess('merged_data.csv', agg_function=agg_function)
    df_grouped_scaled = standardize_data(df_grouped, num_cols)

    # Plot correlation matrix
    plot_correlation_matrix(df_grouped_scaled, num_cols, agg_function=agg_function)

    # Perform PCA
    pca_model, reduced_df, loadings, df = perform_pca(df_grouped_scaled, n_components=4,agg_function=agg_function)

    # List of algorithms to try
    algorithms = [
        {'name': 'KMeans', 'algorithm': KMeans, 'args': {'n_clusters': 2}},
        {'name': 'Hierarchical Clustering', 'algorithm': AgglomerativeClustering, 'args': {'n_clusters': 5}},
        {'name': 'DBSCAN', 'algorithm': DBSCAN, 'args': {'eps': 2, 'min_samples': 5}},
        {'name': 'Gaussian Mixture Models', 'algorithm': GaussianMixture, 'args': {'n_components': 3}},
    ]

    # Try each algorithm
    for alg in algorithms:
        print(f"Running {alg['name']}...")
        clusters = perform_clustering(reduced_df, alg['algorithm'], **alg['args'])
        # Example usage
        koppen_labels = df_grouped_scaled['KOPPEN']  # Extracted before PCA
        generate_biplot(df_grouped_scaled, reduced_df, clusters, pca_model, num_cols, koppen_labels, alg['name'],
                        alg['name'], agg_function)

    # Compute silhouette scores
    models = [
        ("KMeans", KMeans(n_clusters=2)),
        ("GMM", GaussianMixture(n_components=2, random_state=1)),
        ("DBSCAN", DBSCAN(eps=2, min_samples=5)),
        ("Agglomerative", AgglomerativeClustering(n_clusters=5))
    ]

    results = []

    for name, model in models:
        labels = model.fit_predict(reduced_df.values)
        silhouette = silhouette_score(reduced_df.values, labels)
        results.append([name, silhouette])

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=["Model", "Silhouette Score"])

    # Save results to a CSV file
    results_df.to_csv(f'{agg_function}_cat_clustering_results.csv', index=False)


process_and_cluster(agg_function='sum')
process_and_cluster(agg_function='mean')
process_and_cluster(agg_function='min')
process_and_cluster(agg_function='max')
