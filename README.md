FLUXNET Data Processing and Merging with SoilGrids

This repository provides a workflow to process FLUXNET community product data, extracting relevant information and merging it with soil data from the SoilGrids API.

Overview

Workflow

Data Extraction: The script first extracts the raw FLUXNET community product data from zipped files and copies them to a designated directory.

Data Processing: Processes and merges the extracted raw flux data with the associated metadata.

Soil Data Integration: Retrieves and merges soil data from the SoilGrids API based on the site information.

Save to CSV: The merged data, which now includes FLUXNET and SoilGrids data, is saved to a CSV file.

Code Details (Code Directory Only)

extract_zip_files_daily.py: Extracts daily flux data from zipped files and organizes them into the working directory.

fig1_site_map_freq_measurments.py: Generates a site map with frequency measurements for FLUXNET data.

fig2_soil_boxplots_plots.py: Creates boxplots for soil data, visualizing the distribution of key soil properties.

fig3_map_boxplot_and_corr_plot.py: Produces a combined visualization including a site map, boxplots, and correlation plots for data analysis.

get_daily_data_processed.py: Processes daily FLUXNET data, merging it with metadata and formatting it into a structured dataset.

get_year_ranges.py: Extracts and manages the year ranges for data to streamline time-based analyses.

koppen_dict.py: Provides a dictionary that maps site locations to their corresponding Köppen climate classification.

load_flux_data_daily.py: Loads daily FLUXNET data into memory, preparing it for further analysis and processing.

plot_flux_over_time.py: Plots methane flux trends over time for individual or aggregated sites.

rest_api_by_site.py: Interacts with the SoilGrids API to retrieve soil property data for each site location.

rf_soil_all.py: Implements a Random Forest model to analyze the influence of soil properties on methane flux.

rf_soil_climate.py: Uses a Random Forest model to examine the combined effects of soil properties and climate on methane flux.

site_data_spans.py: Identifies the temporal coverage (data spans) for each site to ensure data completeness and consistency.

Pre-requisites

Requirements

Python 3.x

Required Libraries:

pandas

os

shutil

zipfile

requests

Steps to Use

Setup

Ensure that Python is correctly installed and the required libraries are present.

Directory Setup

Download the raw FLUXNET community product data from FLUXNET.

Place the downloaded data in a directory to reference as "path."

Run the Script

Execute the main script to start the data processing workflow:

python get_daily_data_processed.py

Output

After successful execution, the merged data will be saved as merged_data.csv in the working directory.

Troubleshooting

SoilGrids API Issues:

Ensure that the SoilGrids API service is up and running.

Avoid using a VPN that might block API requests.

File Errors:

Ensure the directory paths are correctly set up and accessible.

Confirm that the required zip files are placed in the referenced directory.

