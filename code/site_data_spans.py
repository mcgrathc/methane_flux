import pandas as pd

# Load the data
data = pd.read_csv('merged_data.csv')

# Ensure 'TIMESTAMP' is a datetime type
data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], errors='coerce')

# Extract the date for easier comparison if needed
data['DATE'] = data['TIMESTAMP'].dt.date

# Now let's calculate the time-span for each site
site_date_ranges = data.groupby('SITE_ID')['DATE'].agg([min, max])
site_date_ranges['data_span'] = site_date_ranges['max'] - site_date_ranges['min']

# Sites with more than a year of data
sites_over_year = site_date_ranges[site_date_ranges['data_span'] > pd.Timedelta(days=365)]

# Sites with less than a year of data
sites_under_year = site_date_ranges[site_date_ranges['data_span'] <= pd.Timedelta(days=365)]

# Sites with at least 180 days of data
sites_at_least_half_year = site_date_ranges[site_date_ranges['data_span'] >= pd.Timedelta(days=180)]

# Output the counts
print(f"Number of sites with over a year of data: {len(sites_over_year)}")
print(f"Number of sites with less than a year of data: {len(sites_under_year)}")
print(f"Number of sites with at least 180 days of data: {len(sites_at_least_half_year)}")

# If you want to pull out sites with short periods of data (less than 180 days), you can do:
sites_with_short_data = site_date_ranges[site_date_ranges['data_span'] < pd.Timedelta(days=180)]

# To get a list of site IDs with short data periods
short_data_site_ids = sites_with_short_data.index.tolist()
