import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data
data = pd.read_csv('merged_data.csv')

# Calculate the number of measurements per site
site_measurements = data.groupby(['LAT', 'LON']).size().reset_index(name='num_measurements')

# Create a GeoDataFrame for plotting
gdf = gpd.GeoDataFrame(site_measurements, geometry=gpd.points_from_xy(site_measurements['LON'], site_measurements['LAT']))

# Load a world map for background
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world[world.name != "Antarctica"]

# Plot the map
fig, ax = plt.subplots(figsize=(8, 10), dpi=300)  # High resolution for publication quality
world.plot(ax=ax, color='lightgrey')  # Background world map in light grey

# Define the color map and normalize based on the range of measurements
cmap = plt.cm.Reds  # Use the "Reds" colormap for white-to-red gradient
norm = mcolors.Normalize(vmin=gdf['num_measurements'].min(), vmax=gdf['num_measurements'].max())

# Plot the sites with color representing the number of measurements
sc = ax.scatter(gdf['LON'], gdf['LAT'], c=gdf['num_measurements'], cmap=cmap, norm=norm,
                s=40, edgecolor='black', linewidth=0.3, alpha=0.8)  # Adjusted for publication aesthetics

# Add a smaller color bar with adjusted font size and label position
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.015, pad=0.02)
cbar.set_label('Number of Measurements', fontsize=16, labelpad=10)
cbar.ax.tick_params(labelsize=16)

# Set axis labels with larger font sizes
ax.set_xlabel('Longitude', fontsize=16)
ax.set_ylabel('Latitude', fontsize=16)

# Customize tick label sizes for readability
ax.tick_params(axis='both', which='major', labelsize=16)

# Save the figure as a high-resolution PNG file
plt.savefig('C:\\Users\\mcgr323\\OneDrive - PNNL\\methane_flux\\manuscript\\publication_ready_map.png', dpi=300, bbox_inches='tight', transparent=True)

# Show plot
plt.show()
