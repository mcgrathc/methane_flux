from extract_zip_files_daily import extract_and_copy_fluxnet_data
from load_flux_data_daily import process_and_merge_data
from rest_api_by_site import merge_soil_data_with_sites

# Path to where raw FLUXNET community product is downloaded
path = 'C:\\Users\\mcgr323\\OneDrive - PNNL\\ch4_co2_synthesis\\'

print("Starting data extraction from zipped files...")
# Extracts fluxnet data from zipped files and copies them to a designated directory.
# Note only needs to be run once
extract_and_copy_fluxnet_data(path)
print("Data extraction complete.")

print("Processing and merging flux data with metadata...")
# Processes and merges raw flux data with metadata based on the given directory path.
processed_df = process_and_merge_data(path)
print("Processing and merging complete.")

print("Merging soil data with sites...")
# Merge soil data from SoilGrids API with site information from an input DataFrame.
df_merged = merge_soil_data_with_sites(processed_df)
print("Soil data merge complete.")

# Specify the filename/path
filename = 'merged_data.csv'

print(f"Saving merged data to {filename}...")
# Save df_merged to a CSV file
df_merged = df_merged.rename(columns={'index': 'TIMESTAMP'})
df_merged['TIMESTAMP'] = df_merged['TIMESTAMP'].dt.strftime('%Y/%m/%d')
df_merged.to_csv(filename, index=False)
print(f"Data saved to {filename}!")
