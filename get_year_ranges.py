import pandas as pd
from annual_process_and_clustering_function import load_and_preprocess

df_grouped, num_cols = load_and_preprocess('merged_data.csv', agg_function='sum')
# Convert period to timestamp
df_grouped['TIMESTAMP'] = df_grouped['TIMESTAMP'].dt.to_timestamp()

# Extract year
df_grouped['year'] = pd.to_datetime(df_grouped['TIMESTAMP']).dt.year

# Aggregate min and max years per site
year_ranges = df_grouped.groupby('SITE_NAME')[['year']].agg(['min','max'])

# Pivot min year
min_year = year_ranges.reset_index().pivot(index='SITE_NAME', columns='year', values=('year', 'min'))

# Pivot max year
max_year = year_ranges.reset_index().pivot(index='SITE_NAME', columns='year', values=('year', 'max'))

# Join min and max years into one dataframe
year_ranges = min_year.join(max_year)

# Rename columns
year_ranges.rename(columns={'min':'start_year', 'max':'end_year'}, inplace=True)

# Print first 5 rows
print(year_ranges.head())