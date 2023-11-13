import pandas as pd
import os
import re


def process_and_merge_data(path):
    """
    Processes and merges raw flux data with metadata based on the given directory path.

    Parameters:
    - path (str): The directory path where the flux raw data and metadata files are located.

    Returns:
    - df (DataFrame): The processed and merged DataFrame.
    """

    # Define the directory path for the raw flux data
    csv_dir = f'{path}flux_daily_raw\\'
    df = pd.DataFrame()

    # Iterate through the files in the raw flux data directory
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            # Read each CSV file
            filepath = os.path.join(csv_dir, filename)
            tmp_df = pd.read_csv(filepath)

            # Extract site code from the filename
            site_code = re.search(r'FLX_(.+)_FLUXNET', filename).group(1)

            # Add the extracted site code as a new column to the temporary DataFrame
            tmp_df['SITE_ID'] = site_code

            # Append the temporary DataFrame to the main DataFrame
            df = df.append(tmp_df, ignore_index=True)

    # Read the metadata CSV file
    metadata = pd.read_csv(f'{path}fluxnet_community_raw_data\\FLX_AA-Flx_CH4-META_20201112135337801132.csv')

    # Merge the main DataFrame with the metadata based on the 'SITE_ID' column
    df = df.merge(metadata, on='SITE_ID')

    # Convert the 'TIMESTAMP' column values to datetime format
    fmt = '%Y%m%d'
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format=fmt)

    # Set the 'TIMESTAMP' column as the index of the DataFrame
    df = df.set_index('TIMESTAMP')

    # Select only the relevant columns for the DataFrame
    cols = ['SITE_ID', 'FCH4_F', 'NEE_F', 'H_F', 'LE_F', 'NETRAD_F', 'VPD_F', 'PA_F', 'TA_F',
            'P_F', 'G_F', 'WTD_F', 'SITE_NAME', 'COUNTRY', 'LAT', 'LON',
            'SITE_CLASSIFICATION', 'KOPPEN', 'YEAR_START', 'YEAR_END', 'UTC_OFFSET', 'DOM_VEG']
    df = df[cols]

    # Apply timezone offsets to the 'TIMESTAMP' index
    utc_offsets = df['UTC_OFFSET']
    df.index = df.index.tz_localize('UTC')
    df.index = df.index + pd.to_timedelta(utc_offsets, unit='H')

    # Drop the 'UTC_OFFSET' column
    df.drop(columns=['UTC_OFFSET'], inplace=True)

    # Reset the index of the DataFrame
    df = df.reset_index(drop=False)

    # Handle missing data (-9999.0) across the DataFrame
    original_dtypes = df.dtypes
    df = df.astype(str)
    pattern = r"-9999\.0+"
    df.replace(to_replace=pattern, value='NaN', regex=True, inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(original_dtypes[col])

    # Filter extreme outliers in 'FCH4_F' column
    fch4_percentile = df['FCH4_F'].quantile(0.99)
    df['FCH4_F'] = df['FCH4_F'][df['FCH4_F'].notna() & (df['FCH4_F'] <= fch4_percentile)]

    return df


