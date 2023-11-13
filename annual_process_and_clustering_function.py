import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import linregress
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess(filename, agg_function='min'):
    df = pd.read_csv(filename)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    # Columns for which you want to get the flux
    flux_cols = ['FCH4_F']

    # Columns for which you want to get the mean
    mean_cols = ['P_F', 'TA_F', 'WTD_F', 'PA_F', 'G_F', 'H_F', 'NEE_F',
                 'bdod_0-5cm_mean', 'cec_0-5cm_mean',
                 'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean']

    # Use agg() to specify how you want to aggregate each column
    aggregation_functions = {col: 'mean' for col in mean_cols}
    aggregation_functions.update({col: agg_function for col in flux_cols})

    df_grouped = (df.groupby([df['TIMESTAMP'].dt.to_period('Y'), 'SITE_NAME'])
                  .agg(aggregation_functions)
                  .reset_index())

    # Fill NA values in mean_cols with the mean of the respective columns
    for col in mean_cols:
        df_grouped[col] = df_grouped[col].fillna(df_grouped[col].mean())

    # Fill NA values in flux_cols with the aggregated value
    for col in flux_cols:
        if agg_function == 'sum':
            fill_value = df_grouped[col].sum()
        elif agg_function == 'mean':
            fill_value = df_grouped[col].mean()
        elif agg_function == 'max':
            fill_value = df_grouped[col].max()
        elif agg_function == 'min':
            fill_value = df_grouped[col].min()
        else:
            raise ValueError("Invalid aggregation function")

        df_grouped[col] = df_grouped[col].fillna(fill_value)

    return df_grouped, flux_cols + mean_cols





def standardize_data(df_grouped, num_cols):
    scaler = StandardScaler()
    df_grouped_scaled = df_grouped.copy()
    df_grouped_scaled[num_cols] = scaler.fit_transform(df_grouped[num_cols])
    return df_grouped_scaled


def perform_pca(df_scaled, n_components=4):
    pca = PCA(n_components=n_components)

    # Numerical columns for PCA
    df_scaled = df_scaled.select_dtypes(include=[np.number])

    principalComponents = pca.fit_transform(df_scaled)

    principalDf = pd.DataFrame(data=principalComponents,
                               columns=[f'PC{i}' for i in range(1, n_components + 1)])

    # Explained variance
    explained_variance = pca.explained_variance_ratio_

    # Get the loadings
    loadings = pd.DataFrame(pca.components_.T,
                            columns=[f'PC{i} ({ev:.0%})'
                                     for i, ev in enumerate(explained_variance, 1)],
                            index=df_scaled.columns)

    # Heatmap of loadings
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=2)
    text_size = 30

    ax = sns.heatmap(loadings, annot=True, fmt='.2f', cmap='viridis')

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.tight_layout()

    plt.savefig('annual_sum_loadings_plot.png', dpi=300)

    return pca, principalDf, loadings


def perform_clustering(df_scaled, algorithm, **kwargs):
    """Perform clustering on the data."""
    model = algorithm(**kwargs)
    clusters = model.fit_predict(df_scaled)
    return clusters


def generate_biplot(df, principalDf, clusters, pca, num_cols, cluster_type, algorithm_name):
    # Create dataframe for biplot
    df_pca = pd.concat([principalDf, pd.Series(clusters, name='cluster')], axis=1)

    # Define a color palette suitable for colorblindness
    cud_palette = sns.color_palette("colorblind", n_colors=7)
    cluster_markers = ['o', 's', '^', 'D', 'v', '>', '<']

    # Set larger font sizes
    plt.rcParams.update({'font.size': 16})  # Adjust the font size as needed

    # Generate biplot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")  # Set the style to whitegrid

    # Loop through clusters and set color and marker style based on cluster index
    for i, cluster in enumerate(np.unique(clusters)):
        cluster_data = df_pca[df_pca['cluster'] == cluster]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], color=cud_palette[i], marker=cluster_markers[i], label=f'Cluster {cluster}')

    plt.title(f'Biplot of Soil Data Clusters: {cluster_type}', fontsize=21)  # Larger title font size

    # Add all feature vectors with adjusted scaling for longer vectors
    vectors = pca.components_.T * np.sqrt(pca.explained_variance_) * 4.0  # Adjust the scaling factor for longer vectors

    for i, v in enumerate(vectors):
        plt.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, linewidth=2, color='darkgray')
        # Label the vectors with variable names with slightly increased offsets
        offset_x = 0.08 if v[0] > 0 else -0.08
        offset_y = 0.08 if v[1] > 0 else -0.08
        plt.text(v[0] * 1.2 + offset_x, v[1] * 1.2 + offset_y, num_cols[i], fontsize=14, color='black', ha='center',
                 va='center',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='none'))

    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)', fontsize=21)  # Larger label font size
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)', fontsize=21)  # Larger label font size

    plt.gca().set_facecolor('white')  # Set the background to white

    plt.tight_layout()

    # Create a legend for clusters with larger font size
    plt.legend(fontsize=16)

    # Save the plot with algorithm name in the title
    plt.savefig(f'biplot_ann_sum_{algorithm_name}.png', dpi=300)


def evaluate_clusters(df, algorithms):
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for alg in algorithms:
        if alg['name'] == 'DBSCAN':
            # For DBSCAN, you can use -1 for noise points and positive integers for clusters
            labels = alg['algorithm'](**alg['args']).fit_predict(df)
            silhouette = silhouette_score(df.values, labels)
            db_index = davies_bouldin_score(df.values, labels)
            ch_index = calinski_harabasz_score(df.values, labels)
        else:
            labels = alg['algorithm'](**alg['args']).fit_predict(df)
            silhouette = silhouette_score(df.values, labels)
            db_index = davies_bouldin_score(df.values, labels)
            ch_index = calinski_harabasz_score(df.values, labels)

        silhouette_scores.append([alg['name'], silhouette])
        davies_bouldin_scores.append([alg['name'], db_index])
        calinski_harabasz_scores.append([alg['name'], ch_index])

    return silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores


def plot_cluster_scatter(df, num_cols):
    # Fit clustering model
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(df[num_cols])

    # Add cluster labels as a new column
    df['cluster'] = clusters

    # Create color map from cluster labels
    cluster_colors = {0: 'red', 1: 'blue'}

    # Map cluster labels to colors
    df['cluster_color'] = df['cluster'].map(cluster_colors)

    # Calculate linear regression for each cluster and store the models
    cluster_models = {}
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        slope, intercept, r, p, std_err = linregress(cluster_data['FCH4_F'], cluster_data['soc_0-5cm_mean'])
        r_squared = r ** 2  # Calculate R-squared value
        cluster_models[cluster] = (slope, intercept, r_squared)  # Store R-squared value as well

    x = np.array([df['FCH4_F'].min(), df['FCH4_F'].max()])
    cluster_lines = {}
    for cluster in cluster_models:
        slope, intercept, r_squared = cluster_models[cluster]  # Unpack the R-squared value
        y = intercept + slope * x
        cluster_lines[cluster] = (x, y)

    # Create figure
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
    for i, cluster in enumerate(df['cluster'].unique()):
        # Plot each cluster separately
        axs[i].scatter(df[df['cluster'] == cluster]['FCH4_F'],
                       df[df['cluster'] == cluster]['soc_0-5cm_mean'],
                       c=cluster_colors[cluster], s=100)
        # Add trendline
        x, y = cluster_lines[cluster]
        axs[i].plot(x, y, linewidth=3, color=cluster_colors[cluster])
        # Get R-squared value from the cluster_models dictionary
        r_squared = cluster_models[cluster][2]
        # Labels and titles, including R-squared annotation
        axs[i].set_title(f'Cluster {cluster}', fontsize=18)
        axs[i].set_xlabel('Annual Mean Flux (nmol CH$_4$ m$^{-2}$ s$^{-1}$)', fontsize=18)
        axs[i].set_ylabel('Annual Mean Air Temperature (°C)', fontsize=18)

        # Annotate R-squared value in the top right corner
        axs[i].annotate(f'R² = {r_squared:.2f}', xy=(0.85, 0.9), xycoords='axes fraction', fontsize=12, color='black')

        # Increase the font size of axis ticks
        axs[i].tick_params(axis='both', labelsize=18)

    # General plot adjustments
    plt.tight_layout()
    plt.show()


def analyze_cluster_ranges(df, num_cols, algorithm_name, algorithm):
    # Fit clustering model
    clusters = algorithm.fit_predict(df[num_cols])

    # Add cluster labels as a new column
    df['cluster'] = clusters

    # Create a DataFrame to store cluster statistics
    cluster_stats = pd.DataFrame(columns=['Algorithm', 'Cluster', *num_cols])

    # Calculate and store statistics for each cluster
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        cluster_stats.loc[len(cluster_stats)] = [algorithm_name, cluster, *cluster_data[num_cols].min(), *cluster_data[num_cols].max()]

    return cluster_stats

