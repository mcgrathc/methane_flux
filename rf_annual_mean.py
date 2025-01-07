import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('merged_data.csv')

# Convert 'TIMESTAMP' to datetime
data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], errors='coerce')

# Extract year for grouping
data['YEAR'] = data['TIMESTAMP'].dt.year

# Group by site and year, and calculate mean annual flux for each predictor and target
grouped_data = data.groupby(['SITE_ID', 'YEAR'])[['P_F', 'TA_F', 'WTD_F', 'PA_F', 'G_F', 'H_F', 'NEE_F',
                                                   'bdod_0-5cm_mean', 'cec_0-5cm_mean',
                                                   'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean',
                                                   'FCH4_F']].mean().reset_index()

# Drop rows with missing values
grouped_data.dropna(inplace=True)

# Split the data into features and target
X = grouped_data[['P_F', 'TA_F', 'WTD_F', 'PA_F', 'G_F', 'H_F', 'NEE_F',
                  'bdod_0-5cm_mean', 'cec_0-5cm_mean',
                  'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean']]
y = grouped_data['FCH4_F']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')
