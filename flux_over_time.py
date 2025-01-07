from load_flux_data_daily import process_and_merge_data
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create a directory for the plots if it doesn't exist
output_dir = "flux_plots_overtime"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

path = 'C:\\Users\\mcgr323\\OneDrive - PNNL\\ch4_co2_synthesis\\'

print("Processing and merging flux data with metadata...")
# Processes and merges raw flux data with metadata based on the given directory path.
processed_df = process_and_merge_data(path)
print("Processing and merging complete.")

# Convert the 'index' column to datetime if it's not already
processed_df['index'] = pd.to_datetime(processed_df['index'])

# Set the 'index' column as the DataFrame index
processed_df.set_index('index', inplace=True)

# Loop through each SITE_NAME and plot the data
for site_name in processed_df['SITE_NAME'].unique():
    site_data = processed_df[processed_df['SITE_NAME'] == site_name]

    # Calculate the 95% quantile for FCH4_F and P_F
    fch4_quantile = site_data['WTD_F'].quantile(0.95)
    pf_quantile = site_data['P_F'].quantile(0.95)

    # Start plotting
    plt.figure(figsize=(15, 5))

    # Plot FCH4_F
    plt.plot(site_data.index, site_data['WTD_F'], label='WTD_F', color='blue')
    # Plot P_F
    plt.plot(site_data.index, site_data['P_F'], label='P_F', color='green')

    # Add horizontal lines for the 95% quantiles
    plt.axhline(y=fch4_quantile, color='red', linestyle='--', label=f'95% quantile for FCH4_F')
    plt.axhline(y=pf_quantile, color='purple', linestyle='--', label=f'95% quantile for P_F')

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'FCH4_F and P_F over Time at {site_name}')
    plt.legend()

    # Save the plot
    plt.savefig(f'{output_dir}/{site_name}_flux_over_time.png')
    plt.close()

print(f"All plots are saved in the directory: {output_dir}")


