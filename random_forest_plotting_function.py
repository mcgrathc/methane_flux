import seaborn as sns
import matplotlib.pyplot as plt



def plot_feature_importance(importance_df, agg_function, filename='feature_importance_comparison.png'):
    """
    Plots a colorblind-friendly comparison chart of feature importances.

    Parameters:
    importance_df (DataFrame): A DataFrame containing feature importances for different clusters.
    agg_function (str): The aggregation function to be used ('sum', 'mean', 'min', 'max').
    filename (str, optional): The filename for saving the plot. Defaults to 'feature_importance_comparison.png'.
    """

    # Set a seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Colorblind-friendly colors
    colors = ["#0072B2", "#E69F00"]  # Blue and orange

    # Create the bar chart
    ax = importance_df.plot(kind='barh', figsize=(15, 10), color=colors)

    # Set y-ticks positions
    ax.set_yticks(range(len(importance_df)))

    # Set custom tick labels using 'Model1_Feature' column
    ax.set_yticklabels(importance_df['Cluster 0_Feature'], fontsize=14)

    # Set x and y labels
    plt.xlabel('Importance', fontsize=18)
    plt.ylabel('Feature', fontsize=18)

    # Dictionary to map aggregation function to a more readable format
    agg_function_title_map = {
        'sum': 'Annual Sum of CH$_4$ Flux Feature Importance',
        'mean': 'Annual Mean of CH$_4$ Flux Feature Importance',
        'min': 'Annual Minimum of CH$_4$ Flux Feature Importance',
        'max': 'Annual Maximum of CH$_4$ Flux Feature Importance',
    }
    # Set dynamic title based on aggregation function
    title = agg_function_title_map.get(agg_function, 'Annual CH$_4$ Flux Feature Importance')
    plt.title(title, fontsize=20)

    # Improve legend
    plt.legend(title='Clusters', title_fontsize='16', fontsize='14', loc='upper right')

    # Set tick parameters
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    # Add a light grid for better readability
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.3)
    ax.set_axisbelow(True)

    # Save the figure
    plt.savefig(filename, bbox_inches='tight', dpi=300)

    # Optional: Show the plot
    plt.show()
