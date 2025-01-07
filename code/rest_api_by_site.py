import requests
import pandas as pd


def get_soil_data(site, lat, lon):
    """
    Retrieve soil data from the SoilGrids REST API for a specific site based on latitude and longitude.

    Parameters:
    - site (str): Identifier of the site.
    - lat (float): Latitude of the site.
    - lon (float): Longitude of the site.

    Returns:
    - pd.DataFrame: A DataFrame containing soil data for the specified site.
    """

    # Define the SoilGrids REST API endpoint and parameters
    url = 'https://rest.isric.org/soilgrids/v2.0/properties/query'
    params = {
        'lon': lon,
        'lat': lat,
        'property': ['bdod', 'cec', 'nitrogen', 'phh2o', 'soc'],
        'depth': ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'],
        'value': ['mean', 'uncertainty']
    }

    # Send the API request and get the response
    response = requests.get(url, params=params, headers={'accept': 'application/json'})

    # Check if the response was successful
    if response.status_code != 200:
        print('Error processing response:', response.text)
        return None

    # Convert the response to a Pandas DataFrame
    data = response.json()['properties']

    # Initialize an empty dataframe
    df = pd.DataFrame(columns=['SITE_ID'])

    # Loop through each layer and depth range to populate the dataframe
    for layer in data['layers']:
        for depth in layer['depths']:
            depth_label = depth['label']
            for key, value in depth['values'].items():
                if key == 'mean':
                    column_name = f"{layer['name']}_{depth_label}_mean"
                elif key == 'uncertainty':
                    column_name = f"{layer['name']}_{depth_label}_uncertainty"
                else:
                    continue
                df[column_name] = [value]

    # Add the site ID to the dataframe
    df['SITE_ID'] = site

    return df


def merge_soil_data_with_sites(df_sites):
    """
    Merge soil data from SoilGrids API with site information from an input DataFrame.

    Parameters:
    - df_sites (pd.DataFrame): DataFrame containing site information with columns "SITE_ID", "LAT", and "LON".

    Returns:
    - pd.DataFrame: Merged DataFrame containing site information and soil data.
    """

    # Create an empty DataFrame to hold the soil data for all sites
    soil_data_all = pd.DataFrame()

    sites = df_sites["SITE_ID"].unique()
    for site in sites:
        lat = df_sites.loc[df_sites["SITE_ID"] == site, "LAT"].iloc[0]
        lon = df_sites.loc[df_sites["SITE_ID"] == site, "LON"].iloc[0]
        soil_data = get_soil_data(site, lat, lon)
        soil_data_all = pd.concat([soil_data_all, soil_data], ignore_index=True)

    # Merge the soil data with the input site information DataFrame
    df_merged = pd.merge(df_sites, soil_data_all, on="SITE_ID")

    return df_merged


