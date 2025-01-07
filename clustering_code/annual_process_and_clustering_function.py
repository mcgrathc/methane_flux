import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess(filename, agg_function='sum'):
    """
    Load a CSV file and preprocess it for data analysis.

    This function reads a CSV file, converts the 'TIMESTAMP' column to datetime,
    and aggregates data based on specified columns. It handles two types of columns:
    flux columns and mean columns. For flux columns, data can be aggregated using a specified
    function ('sum', 'mean', 'max', 'min'). For mean columns, the mean is used for aggregation.
    NA values are filled with the aggregated values in their respective columns.

    Parameters:
    filename (str): The path to the CSV file to be loaded.
    agg_function (str, optional): The aggregation function to be applied to flux columns.
                                  Defaults to 'sum'. Options include 'sum', 'mean', 'max', 'min'.

    Returns:
    tuple: A tuple containing the following elements:
        - DataFrame: The preprocessed and aggregated DataFrame.
        - list: A list of column names that were used for flux and mean calculations.
    """

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
    """
    Standardize numerical columns in a DataFrame using StandardScaler.

    This function takes a DataFrame and a list of column names representing numerical data.
    It applies standardization to these columns, transforming them such that they have a mean of 0
    and a standard deviation of 1. This is done using the StandardScaler from sklearn.preprocessing.
    The standardized data is returned in a new DataFrame, preserving the original DataFrame.

    Parameters:
    df_grouped (DataFrame): The DataFrame containing the data to be standardized.
    num_cols (list of str): A list of column names in the DataFrame representing numerical data
                               that needs to be standardized.

    Returns:
    DataFrame: A copy of the input DataFrame with the specified numerical columns standardized.
    """

    scaler = StandardScaler()
    df_grouped_scaled = df_grouped.copy()
    df_grouped_scaled[num_cols] = scaler.fit_transform(df_grouped[num_cols])

    return df_grouped_scaled


def plot_correlation_matrix(df, num_cols, agg_function='sum'):
    """
    Generates and saves a heatmap of the correlation matrix for the specified numeric columns in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    num_cols (list): List of numeric column names in the DataFrame for which the correlation matrix is to be computed.
    agg_function (str, optional): Aggregation function used in data preprocessing, included in the filename of the saved plot. Defaults to 'sum'.
    """

    # Compute the correlation matrix
    corr_matrix = df[num_cols].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set the font scale for larger text
    sns.set(font_scale=1.2)  # Increase the scale for larger annotation text

    # Create the correlation plot
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, mask=mask, vmin=-1, vmax=1)

    # Remove grid lines and set background color to white
    ax.grid(False)
    ax.set_facecolor('white')  # Set the background color to white

    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{agg_function}_correlation_plot.png', dpi=300)


def perform_pca(df_scaled, n_components=4, agg_function='sum'):
    """
    Perform Principal Component Analysis (PCA) on the given DataFrame.

    This function applies PCA to a scaled DataFrame, reducing its dimensions to the number of
    principal components specified. It returns the PCA model, a DataFrame containing the transformed
    principal components, and the loadings for each component. Additionally, a heatmap of the loadings
    is generated and saved as a PNG file, indicating the contribution of each feature to the principal components.

    Parameters:
    df_scaled (DataFrame): The scaled DataFrame on which PCA is to be performed. This DataFrame should contain only numerical columns.
    n_components (int, optional): The number of principal components to keep. Defaults to 4.
    agg_function (str, optional): The aggregation function used in data preprocessing, which will be part of the filename of the saved heatmap. Defaults to 'sum'.

    Returns:
    tuple: A tuple containing the following elements:
        - PCA object: The fitted PCA model.
        - DataFrame: A DataFrame with the principal components.
        - DataFrame: A DataFrame containing the loadings of the features on each principal component.
    """
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

    ax = sns.heatmap(loadings, annot=True, fmt='.2f', cmap='viridis', vmin=-1, vmax=1)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.tight_layout()

    plt.savefig(f'{agg_function}_annual_loadings_plot.png', dpi=300)

    return pca, principalDf, loadings


def perform_clustering(df_scaled, algorithm, **kwargs):
    """
    Perform clustering on a scaled DataFrame using a specified algorithm.

    This function applies a clustering algorithm to the given DataFrame. The algorithm
    and its parameters are specified by the user. It fits the algorithm to the data and
    predicts the cluster for each instance in the DataFrame. The function returns an array
    of cluster labels corresponding to each row in the DataFrame.

    Parameters:
    df_scaled (DataFrame): The scaled DataFrame to be clustered. This should be a DataFrame
                           where features are scaled appropriately for the clustering algorithm.
    algorithm (class): The clustering algorithm class from a library like sklearn.
                       For example, sklearn.cluster.KMeans.
    **kwargs: Additional keyword arguments specific to the clustering algorithm being used.
              These arguments are passed directly to the algorithm.

    Returns:
    ndarray: An array of cluster labels for each instance in the DataFrame.
    """
    model = algorithm(**kwargs)
    clusters = model.fit_predict(df_scaled)
    return clusters


def generate_biplot(df, principalDf, clusters, pca, num_cols, cluster_type, algorithm_name, agg_function = 'sum'):
    """
    Generate a biplot for visualizing the clusters in the principal component space.

    This function creates a biplot of the first two principal components, showing how different instances
    (rows in the DataFrame) are grouped into clusters. The biplot also displays vectors representing the
    original features, giving an indication of their contribution to the principal components. Clusters are
    visualized with different colors and markers for clarity. The function saves the generated biplot as a PNG file.

    Parameters:
    df (DataFrame): Original DataFrame before scaling and PCA. Used for metadata and labels.
    principalDf (DataFrame): DataFrame containing principal component values for each instance.
    clusters (array-like): Array of cluster labels for each instance.
    pca (PCA): The PCA model used for dimensionality reduction.
    num_cols (list of str): List of column names representing the original numerical features.
    cluster_type (str): The type of clustering performed (e.g., 'KMeans', 'DBSCAN').
    algorithm_name (str): The name of the clustering algorithm used.
    agg_function (str, optional): The aggregation function used in data preprocessing. This is included in the filename of the saved plot. Defaults to 'sum'.

    Returns:
    None: This function does not return a value but saves the biplot as a PNG file.
    """

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
    plt.savefig(f'manuscript\\{agg_function}_biplot_ann_{algorithm_name}.png', dpi=300)
