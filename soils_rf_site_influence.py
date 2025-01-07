def compare_models_with_and_without_soil(data, site_id):
    site_data = data[data['SITE_ID'] == site_id]
    columns_to_group = ['P_F', 'TA_F', 'WTD_F', 'NEE_F', 'FCH4_F']
    soil_columns = ['bdod_0-5cm_mean', 'cec_0-5cm_mean', 'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean']

    # Group and preprocess data
    grouped_data = site_data.groupby(['YEAR'])[columns_to_group + soil_columns].mean().reset_index()
    grouped_data.dropna(inplace=True)

    # Ensure there are enough samples for splitting
    if len(grouped_data) < 2:
        return None

    X_columns = [col for col in columns_to_group if col != 'FCH4_F']
    X = grouped_data[X_columns]
    y = grouped_data['FCH4_F']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate models with and without soil properties
    results = {}
    for include_soil in [True, False]:
        if include_soil:
            X_train_soil = X_train.join(grouped_data[soil_columns])
            X_test_soil = X_test.join(grouped_data[soil_columns])
            model = RandomForestRegressor(random_state=42).fit(X_train_soil, y_train)
            y_pred = model.predict(X_test_soil)
        else:
            model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
            y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[include_soil] = {'RMSE': rmse, 'R2': r2}

    # Return the difference in performance metrics
    diff_rmse = results[True]['RMSE'] - results[False]['RMSE']
    diff_r2 = results[True]['R2'] - results[False]['R2']
    return {'Site': site_id, 'Diff_RMSE': diff_rmse, 'Diff_R2': diff_r2}


# Main analysis
data = pd.read_csv('merged_data.csv')
data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], errors='coerce')
data['YEAR'] = data['TIMESTAMP'].dt.year

# Climate boxplot
# Köppen Description Dictionary
koppen_descriptions = {
    'Dfc': 'Continental',
    'Cfb': 'Temperate',
    'Csa': 'Temperate',
    'Cfa': 'Temperate',
    'Dfb': 'Continental',
    'Dfa': 'Continental',
    'Dwc': 'Continental',
    'ET': 'Polar',
    'Am': 'Tropical',
    'Cwa': 'Temperate',
    'Aw': 'Tropical',
    'Dfd': 'Continental',
    'Af': 'Tropical',
    'Cwc': 'Temperate',
    'Dwa': 'Continental',
    'Bsh': 'Arid'
}

# Map the 'KOPPEN' column to climate categories using the koppen_descriptions dictionary
data['CLIMATE'] = data['KOPPEN'].map(koppen_descriptions)

# Remove Arid Climates
data = data[data['CLIMATE'] != 'Arid']

site_influences = []
for site_id in data['SITE_ID'].unique():
    site_influence = compare_models_with_and_without_soil(data, site_id)
    if site_influence is not None:
        site_influences.append(site_influence)

influence_df = pd.DataFrame(site_influences).sort_values(by='Diff_R2', ascending=False)

# Save the DataFrame as a CSV file
influence_df.to_csv('site_influences.csv', index=False)
