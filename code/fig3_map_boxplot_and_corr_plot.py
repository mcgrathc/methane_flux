import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mcolors

# Load the data
data = pd.read_csv('merged_data.csv')
df = pd.DataFrame(data)

# Convert 'TIMESTAMP' to datetime if it's not already
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')

# Extract the day for grouping by annual data
df['DATE'] = df['TIMESTAMP'].dt.date

# Climate boxplot
# Köppen Description Dictionary
koppen_descriptions = {
    'Dfc': 'Continental',
    'Cfb': 'Temperate',
    'Csa': 'Temperate',
    'Cfa': 'Temperate',
    'Dfb': 'Continental',
    'Dfa': 'Continental',
    'Dwc': 'Continental',
    'ET': 'Polar',
    'Am': 'Tropical',
    'Cwa': 'Temperate',
    'Aw': 'Tropical',
    'Dfd': 'Continental',
    'Af': 'Tropical',
    'Cwc': 'Temperate',
    'Dwa': 'Continental',
    'Bsh': 'Arid'
}

# Map the 'KOPPEN' column to climate categories using the koppen_descriptions dictionary
df['CLIMATE'] = df['KOPPEN'].map(koppen_descriptions)

# Remove Arid Climates
df = df[df['CLIMATE'] != 'Arid']

# Get unique classifications
classifications = df['SITE_CLASSIFICATION'].unique()

# Sort the classifications if necessary
classifications = np.sort(classifications)

# Define a custom palette with more distinct colors
custom_palette = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080"
]

# Create a colormap from the custom palette
colormap = ListedColormap(custom_palette)

# Create a GeoDataFrame from the DataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['LON'], df['LAT']))

# Plot the world map without Antarctica
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world[world.name != "Antarctica"]

# Create a figure with GridSpec layout
fig = plt.figure(figsize=(30, 10))
gs = GridSpec(2, 2, height_ratios=[2, 2], width_ratios=[2, 1])

# Plot the world map with enhanced aesthetics
ax0 = fig.add_subplot(gs[:, 0])
world.plot(ax=ax0, color='lightgrey')
gdf.plot(ax=ax0, column='SITE_CLASSIFICATION', markersize=200, legend=False, cmap=colormap,
         edgecolor='black', linewidth=1)
ax0.set_aspect('equal')

# Create a legend with custom patches
legend_patches = [mpatches.Patch(color=colormap.colors[i], label=classifications[i]) for i in range(len(classifications))]
leg = ax0.legend(handles=legend_patches, title='Site Classification', fontsize=14, loc='lower left')
leg.set_title('Site Classification', prop={'size': 16})
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_edgecolor("black")

# Set axis labels with font size 22
ax0.set_xlabel('Longitude', fontsize=22)
ax0.set_ylabel('Latitude', fontsize=22)

# Increase the font size of the tick labels for both axes to 22
ax0.tick_params(axis='x', labelsize=22)
ax0.tick_params(axis='y', labelsize=22)

# Calculate the daily means for each SITE_CLASSIFICATION
daily_means_wetland = df.groupby(['DATE', 'SITE_CLASSIFICATION'])['FCH4_F'].mean().reset_index()

# Plot the boxplot for SITE_CLASSIFICATION
ax1 = fig.add_subplot(gs[0, 1])
sns.boxplot(x='FCH4_F', y='SITE_CLASSIFICATION', data=daily_means_wetland,
            palette=dict(zip(classifications, custom_palette)), orient='h', ax=ax1)
ax1.set_xlabel('Daily Mean Flux (nmol CH₄ m⁻² s⁻¹)', fontsize=22)
ax1.set_ylabel('Site Classification', fontsize=22)
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

# Calculate the daily means for each CLIMATE
daily_means_climate = df.groupby(['DATE', 'CLIMATE'])['FCH4_F'].mean().reset_index()

# Define a custom palette for the climate categories
climate_palette = [
    "#1f77b4",  # Continental
    "#ff7f0e",  # Temperate
    "#2ca02c",  # Polar
    "#d62728",  # Tropical
]

# Plot the boxplot for CLIMATE
ax2 = fig.add_subplot(gs[1, 1])
sns.boxplot(x='FCH4_F', y='CLIMATE', data=daily_means_climate,
            palette=dict(zip(df['CLIMATE'].unique(), climate_palette)), orient='h', ax=ax2)
ax2.set_xlabel('Daily Mean Flux (nmol CH₄ m⁻² s⁻¹)', fontsize=22)
ax2.set_ylabel('Climate Category', fontsize=22)
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('manuscript/combined_figure.png', dpi=300, bbox_inches='tight')


### Correlation plot ###
# Filter the columns that have "_0-5cm_mean" in the header
filtered_df = df.filter(regex='_0-5cm_mean')

# Rename the columns for better readability
renamed_columns = {
    'bdod_0-5cm_mean': 'Bd',
    'cec_0-5cm_mean': 'CEC',
    'phh2o_0-5cm_mean': 'pH',
    'soc_0-5cm_mean': 'SOC',
    'nitrogen_0-5cm_mean': 'TN'
}
filtered_df.rename(columns=renamed_columns, inplace=True)

# Compute the correlation matrix
corr_matrix = filtered_df.corr()

# Remove the upper triangle of the correlation matrix to avoid redundancy
corr_matrix = corr_matrix.where(~np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Stack the matrix and reset the index
corr_matrix = corr_matrix.stack().reset_index()
corr_matrix.columns = ['Variable1', 'Variable2', 'Correlation']

# Remove 1:1 correlations
corr_matrix = corr_matrix[corr_matrix['Variable1'] != corr_matrix['Variable2']]

# Create a new column for the y-axis labels that combines the variable names
corr_matrix['Pair'] = corr_matrix[['Variable1', 'Variable2']].apply(lambda x: f'{x[0]} & {x[1]}', axis=1)

# Sort by correlation
corr_matrix.sort_values('Correlation', inplace=True)

# Use the 'coolwarm' colormap for the gradient
cmap = sns.color_palette("coolwarm", as_cmap=True)

# Normalize the Correlation column to map the colors
norm = mcolors.Normalize(vmin=corr_matrix['Correlation'].min(), vmax=corr_matrix['Correlation'].max())

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
# Color the bars with the normalized correlation values
bar_colors = [cmap(norm(value)) for value in corr_matrix['Correlation']]
barplot = sns.barplot(data=corr_matrix, x='Correlation', y='Pair', palette=bar_colors)

# Annotate the bars with the correlation coefficients
for p in barplot.patches:
    barplot.annotate(f'{p.get_width():.2f}',
                     (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2.),
                     ha='right' if p.get_width() > 0 else 'left',
                     va='center',
                     xytext=(-5 if p.get_width() > 0 else 5, 0),
                     textcoords='offset points',
                     fontsize=20)  # Increase the font size here

# Set the labels and title with increased font sizes for readability
ax.set_xlabel('Correlation Coefficient', fontsize=18)
ax.set_ylabel('Soil Property Pairs', fontsize=18)

# Increase the font size of the tick labels for both axes
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# Invert y-axis to have the highest correlations at the top
ax.invert_yaxis()

# Tight layout to ensure everything fits without overlapping
plt.tight_layout()

# Save the figure with high resolution
plt.savefig('manuscript/soil_properties_correlation.png', dpi=300, bbox_inches='tight')



