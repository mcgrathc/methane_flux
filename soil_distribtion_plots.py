import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('merged_data.csv')
data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], errors='coerce')
data['YEAR'] = data['TIMESTAMP'].dt.year
data['MONTH'] = data['TIMESTAMP'].dt.month

# Convert SOC from dg/kg to g/kg
data['soc_0-5cm_mean'] = data['soc_0-5cm_mean'] / 10

# Convert pH from pHx10 to conventional pH
data['phh2o_0-5cm_mean'] = data['phh2o_0-5cm_mean'] / 10

# Convert bdod and cec to conventional units
data['bdod_0-5cm_mean'] = data['bdod_0-5cm_mean'] / 100  # cg/cm³ to g/cm³
data['cec_0-5cm_mean'] = data['cec_0-5cm_mean'] / 10  # cmol(+)/kg x10 to cmol(+)/kg

# Define bins for SOC, pH, WTD, temperature, bdod, and cec
soc_bins = [0, 50, 100, 150, 200, 450]
ph_bins = [4.0, 5.0, 5.5, 6.0, 6.5, 7.5]
wtd_bins = [-2, -1, 0, 1, 2]
temperature_bins = [-20, 0, 10, 20, 30, 40]
bdod_bins = [0, 0.25, 0.5, 1, 1.5]  # Correct bins for bdod in g/cm³
cec_bins = [9, 20, 40, 60, 900]  # Correct bins for cec in cmol(+)/kg

# Create bin columns
data['soc_bin'] = pd.cut(data['soc_0-5cm_mean'], bins=soc_bins)
data['ph_bin'] = pd.cut(data['phh2o_0-5cm_mean'], bins=ph_bins)
data['wtd_bin'] = pd.cut(data['WTD_F'], bins=wtd_bins)
data['temperature_bin'] = pd.cut(data['TA_F'], bins=temperature_bins)
data['bdod_bin'] = pd.cut(data['bdod_0-5cm_mean'], bins=bdod_bins)
data['cec_bin'] = pd.cut(data['cec_0-5cm_mean'], bins=cec_bins)

# Apply log10 scale to FCH4 values (handling non-positive values)
data['FCH4_log10'] = np.log10(data['FCH4_F'].where(data['FCH4_F'] > 0))

# Function to format bin labels with or without decimals
def format_bins(bin_labels, variable):
    if variable in ['ph_bin', 'bdod_bin']:
        # Format with one decimal place for pH and BDOD
        return [f"{interval.left:.1f}-{interval.right:.1f}" for interval in bin_labels]
    else:
        # Format without decimal places for other variables
        return [f"{int(interval.left)}-{int(interval.right)}" for interval in bin_labels]

# Set up figure and subplots (3 columns and 2 rows)
fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

# Common font sizes
label_fontsize = 14
tick_fontsize = 12
annotation_fontsize = 18

# Define annotation labels
annotations = ['a', 'b', 'c', 'd', 'e', 'f']

# Plot SOC Box Plot
axs[0, 0].boxplot([data[data['soc_bin'] == b]['FCH4_log10'].dropna() for b in data['soc_bin'].cat.categories],
                  patch_artist=True, boxprops=dict(facecolor='#2E86C1'))
axs[0, 0].set_xlabel('SOC (g/kg)', fontsize=label_fontsize)
axs[0, 0].set_ylabel('Monthly Mean FCH$_4$ (nmol CH₄ m⁻² s⁻¹)', fontsize=label_fontsize)
axs[0, 0].set_xticklabels(format_bins(data['soc_bin'].cat.categories, 'soc_bin'), fontsize=tick_fontsize, rotation=0)
axs[0, 0].text(-0.3, 1.05, annotations[0], transform=axs[0, 0].transAxes,
               fontsize=annotation_fontsize, fontweight='bold', va='top', ha='right')

# Plot pH Box Plot
axs[0, 1].boxplot([data[data['ph_bin'] == b]['FCH4_log10'].dropna() for b in data['ph_bin'].cat.categories],
                  patch_artist=True, boxprops=dict(facecolor='#28B463'))
axs[0, 1].set_xlabel('pH', fontsize=label_fontsize)
axs[0, 1].set_ylabel('Monthly Mean FCH$_4$ (nmol CH₄ m⁻² s⁻¹)', fontsize=label_fontsize)
axs[0, 1].set_xticklabels(format_bins(data['ph_bin'].cat.categories, 'ph_bin'), fontsize=tick_fontsize, rotation=0)
axs[0, 1].text(-0.3, 1.05, annotations[1], transform=axs[0, 1].transAxes,
               fontsize=annotation_fontsize, fontweight='bold', va='top', ha='right')

# Plot WTD Box Plot
axs[0, 2].boxplot([data[data['wtd_bin'] == b]['FCH4_log10'].dropna() for b in data['wtd_bin'].cat.categories],
                  patch_artist=True, boxprops=dict(facecolor='#E74C3C'))
axs[0, 2].set_xlabel('WTD (m)', fontsize=label_fontsize)
axs[0, 2].set_ylabel('Monthly Mean FCH$_4$ (nmol CH₄ m⁻² s⁻¹)', fontsize=label_fontsize)
axs[0, 2].set_xticklabels(format_bins(data['wtd_bin'].cat.categories, 'wtd_bin'), fontsize=tick_fontsize, rotation=0)
axs[0, 2].text(-0.3, 1.05, annotations[2], transform=axs[0, 2].transAxes,
               fontsize=annotation_fontsize, fontweight='bold', va='top', ha='right')

# Plot Temperature Box Plot
axs[1, 0].boxplot([data[data['temperature_bin'] == b]['FCH4_log10'].dropna() for b in data['temperature_bin'].cat.categories],
                  patch_artist=True, boxprops=dict(facecolor='#F39C12'))
axs[1, 0].set_xlabel('Temperature (°C)', fontsize=label_fontsize)
axs[1, 0].set_ylabel('Monthly Mean FCH$_4$ (nmol CH₄ m⁻² s⁻¹)', fontsize=label_fontsize)
axs[1, 0].set_xticklabels(format_bins(data['temperature_bin'].cat.categories, 'temperature_bin'), fontsize=tick_fontsize, rotation=0)
axs[1, 0].text(-0.3, 1.05, annotations[3], transform=axs[1, 0].transAxes,
               fontsize=annotation_fontsize, fontweight='bold', va='top', ha='right')

# Plot BDOD Box Plot
axs[1, 1].boxplot([data[data['bdod_bin'] == b]['FCH4_log10'].dropna() for b in data['bdod_bin'].cat.categories],
                  patch_artist=True, boxprops=dict(facecolor='#8E44AD'))
axs[1, 1].set_xlabel('BDOD (g/cm³)', fontsize=label_fontsize)
axs[1, 1].set_ylabel('Monthly Mean FCH$_4$ (nmol CH₄ m⁻² s⁻¹)', fontsize=label_fontsize)
axs[1, 1].set_xticklabels(format_bins(data['bdod_bin'].cat.categories, 'bdod_bin'), fontsize=tick_fontsize, rotation=0)
axs[1, 1].text(-0.3, 1.05, annotations[4], transform=axs[1, 1].transAxes,
               fontsize=annotation_fontsize, fontweight='bold', va='top', ha='right')

# Plot CEC Box Plot
axs[1, 2].boxplot([data[data['cec_bin'] == b]['FCH4_log10'].dropna() for b in data['cec_bin'].cat.categories],
                  patch_artist=True, boxprops=dict(facecolor='#D35400'))
axs[1, 2].set_xlabel('CEC (cmol(+)/kg)', fontsize=label_fontsize)
axs[1, 2].set_ylabel('Monthly Mean FCH$_4$ (nmol CH₄ m⁻² s⁻¹)', fontsize=label_fontsize)
axs[1, 2].set_xticklabels(format_bins(data['cec_bin'].cat.categories, 'cec_bin'), fontsize=tick_fontsize, rotation=0)
axs[1, 2].text(-0.3, 1.05, annotations[5], transform=axs[1, 2].transAxes,
               fontsize=annotation_fontsize, fontweight='bold', va='top', ha='right')

# Adjust aesthetics
for ax in axs.flat:
    ax.grid(visible=True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Save the plot as a high-resolution image
plt.savefig('manuscript//annotated_boxplots_cleaned_ranges.png', dpi=300, bbox_inches='tight')

# Show the plots
plt.show()

