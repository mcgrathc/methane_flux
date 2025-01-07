import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from clustering_code.annual_process_and_clustering_function import standardize_data, perform_pca, load_and_preprocess


# Define a function to find the optimal K for KMeans and K-Medoids
def find_optimal_k(data, num_cols):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(data[num_cols])
        distortions.append(
            sum(np.min(cdist(data[num_cols], kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data[num_cols].shape[
                0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


# Define a function to find the optimal number of components for GMM
def find_optimal_n_components(data, num_cols):
    n_components_list = range(2, 5)
    silhouette_scores = []
    for n in n_components_list:
        gmm = GaussianMixture(n_components=n, random_state=1)
        labels = gmm.fit_predict(data[num_cols])
        score = silhouette_score(data[num_cols], labels)
        silhouette_scores.append(score)

    # Plot silhouette scores vs n_components
    plt.plot(n_components_list, silhouette_scores, 'bo-')
    plt.xlabel('Number of components')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette Scores for Different Numbers of Components (GMM)')
    plt.show()


# Define a function to find the optimal parameters for DBSCAN
def find_optimal_eps_min_samples(data, num_cols):
    eps_values = [0.5, 1, 1.5, 2]
    min_samples_values = [5, 10, 15, 20]

    best_eps = 0
    best_min_samples = 0
    best_silhouette_score = -1

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(data[num_cols])

            if len(np.unique(clusters)) == 1:
                continue

            silhouette = silhouette_score(data[num_cols], clusters)

            if silhouette > best_silhouette_score:
                best_silhouette_score = silhouette
                best_eps = eps
                best_min_samples = min_samples

    print("Best eps:", best_eps)
    print("Best min_samples:", best_min_samples)
    print("Best silhouette score:", best_silhouette_score)


# Define a function to plot dendrograms for hierarchical clustering
def plot_dendrograms(data, num_cols):
    methods = ['single', 'average', 'complete']

    fig, axs = plt.subplots(len(methods), 1, figsize=(20, 15))

    for i, method in enumerate(methods):
        Z = linkage(data[num_cols], metric='euclidean', method=method)

        dendrogram(Z, ax=axs[i])

        axs[i].set_title(f'Dendrogram ({method.capitalize()} Linkage)')
        axs[i].set_ylabel('Distance')

    plt.tight_layout()
    plt.show()


# Load and pre-process data
df_grouped, num_cols = load_and_preprocess('merged_data.csv', agg_function='max')

# Standardize the data
df_grouped_scaled = standardize_data(df_grouped, num_cols)

# Perform PCA on the scaled data
pca_model, reduced_df, loadings = perform_pca(df_grouped_scaled)


find_optimal_k(df_grouped_scaled, num_cols)
find_optimal_n_components(df_grouped_scaled, num_cols)
find_optimal_eps_min_samples(df_grouped_scaled, num_cols)
plot_dendrograms(df_grouped_scaled, num_cols)
