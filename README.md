## FLUXNET Data Processing and Merging with SoilGrids
This repository provides a workflow to process FLUXNET community product data, extracting relevant information and merging it with soil data from the SoilGrids API.

### Overview
Data Extraction: The script first extracts the raw FLUXNET community product data from zipped files and copies them to a designated directory.
Data Processing: Processes and merges the extracted raw flux data with the associated metadata.
Soil Data Integration: Retrieves and merges soil data from the SoilGrids API based on the site information.
Save to CSV: The merged data, which now includes FLUXNET and SoilGrids data, is saved to a CSV file.

### Pre-requisites
Python 3.x
Required Libraries: pandas, os, shutil, zipfile, requests

## Steps to Use
### Setup: Ensure that Python is correctly installed and the required libraries are present.

### Directory Setup: 
Download the raw FLUXNET community product data from https://fluxnet.org/data/fluxnet-ch4-community-product/ and place it in a directory to reference as "path"

### Run the Script: 
Execute the main script to start the data processing workflow:

*python get_daily_data_processed.py*

### Output: 
After successful execution, the merged data will be saved as merged_data.csv in the working directory.

### Troubleshooting
If you encounter any issues with the SoilGrids API, ensure that the service is up and running and you are not on any VPN.
Ensure the directory paths are correctly set up and accessible to avoid file-related errors.
