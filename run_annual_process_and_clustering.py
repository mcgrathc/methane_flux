import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from tabulate import tabulate
from annual_process_and_clustering_function import load_and_preprocess, standardize_data, perform_pca, perform_clustering, generate_biplot

df_grouped, num_cols = load_and_preprocess('merged_data.csv', agg_function='sum')
df_grouped_scaled = standardize_data(df_grouped, num_cols)

# Compute the correlation matrix
corr_matrix = df_grouped_scaled[num_cols].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set the style to "whitegrid"
sns.set_style("whitegrid")

# Create the correlation plot
plt.figure(figsize=(12, 8))
text_size = 10
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, mask=mask, annot_kws={"size": text_size})
plt.title('Correlation Plot')
plt.tight_layout()
plt.savefig('annual_max_correlation_plot.png', dpi = 300)
plt.show()

# Assuming df_scaled is your scaled dataframe
pca_model, reduced_df, loadings = perform_pca(df_grouped_scaled)

# List of algorithms to try
algorithms = [
    {'name': 'KMeans', 'algorithm': KMeans, 'args': {'n_clusters': 2}},
    {'name': 'Hierarchical Clustering', 'algorithm': AgglomerativeClustering, 'args': {'n_clusters': 5}},
    {'name': 'DBSCAN', 'algorithm': DBSCAN, 'args': {'eps': 2, 'min_samples': 10}},
    {'name': 'Gaussian Mixture Models', 'algorithm': GaussianMixture, 'args': {'n_components': 2}},
]

# Try each algorithm
for alg in algorithms:
    print(f"Running {alg['name']}...")
    clusters = perform_clustering(reduced_df, alg['algorithm'], **alg['args'])
    generate_biplot(df_grouped_scaled, reduced_df, clusters, pca_model, num_cols, alg['name'], alg['name'])

models = [
    ("KMeans", KMeans(n_clusters=2)),
    ("GMM", GaussianMixture(n_components=2, random_state=1)),
    ("DBSCAN", DBSCAN(eps=2, min_samples=10)),
    ("Agglomerative", AgglomerativeClustering(n_clusters=5))
]

results = []

for name, model in models:
  labels = model.fit_predict(reduced_df.values)
  silhouette = silhouette_score(reduced_df.values, labels)
  results.append([name, silhouette])

headers = ["Model", "Silhouette Score"]
print(tabulate(results, headers=headers, tablefmt="grid"))


# Fit clustering model
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(reduced_df)

# Add cluster labels as a new column
df_grouped['cluster'] = clusters

# Create color map from cluster labels
cluster_colors = {0: 'blue', 1: 'gold'}

# Map cluster labels to colors
df_grouped['cluster_color'] = df_grouped['cluster'].map(cluster_colors)

# Calculate linear regression for each cluster and store the models
cluster_models = {}
for cluster in df_grouped['cluster'].unique():
    cluster_data = df_grouped[df_grouped['cluster'] == cluster]
    slope, intercept, r, p, std_err = linregress(cluster_data['FCH4_F'], cluster_data['TA_F'])
    r_squared = r ** 2  # Calculate R-squared value
    cluster_models[cluster] = (slope, intercept, r_squared)  # Store R-squared value as well

x = np.array([df_grouped['FCH4_F'].min(), df_grouped['FCH4_F'].max()])
cluster_lines = {}
for cluster in cluster_models:
    slope, intercept, _ = cluster_models[cluster]  # Unpack the R-squared value as well
    y = intercept + slope * x
    cluster_lines[cluster] = (x, y)

# Create figure
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
for i, cluster in enumerate(df_grouped['cluster'].unique()):
    # Plot each cluster separately
    axs[i].scatter(df_grouped[df_grouped['cluster'] == cluster]['FCH4_F'],
                   df_grouped[df_grouped['cluster'] == cluster]['TA_F'],
                   c=cluster_colors[cluster], s=100)
    # Add trendline
    x, y = cluster_lines[cluster]
    axs[i].plot(x, y, linewidth=3, color=cluster_colors[cluster])
    # Get R-squared value from the cluster_models dictionary
    r_squared = cluster_models[cluster][2]
    # Labels and titles, including R-squared annotation
    axs[i].set_title(f'Cluster {cluster} ', fontsize=24)
    if i == 0:
        axs[i].set_xlabel('Annual Accumulated Flux (nmol CH$_4$ m$^{-2}$ s$^{-1}$)', fontsize=24)
    axs[i].set_ylabel('Annual Mean Air Temperature (°C)', fontsize=24)

    # Increase the font size of axis ticks
    axs[i].tick_params(axis='both', labelsize=18)

    # Add log10 scale to the 'FCH4_F' and 'soc_0-5cm_mean' axes
    axs[i].set_xscale('log')
    #axs[i].set_yscale('log')

# General plot adjustments
plt.tight_layout()
plt.show()
