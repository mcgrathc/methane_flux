from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from annual_process_and_clustering_function import load_and_preprocess, standardize_data, perform_pca
from rf_function import train_and_visualize_rf_with_hyperparam_tuning

df_grouped, num_cols = load_and_preprocess('merged_data.csv', agg_function='sum')
df_grouped_scaled = standardize_data(df_grouped, num_cols)

# Assuming df_scaled is your scaled dataframe
pca_model, reduced_df, loadings = perform_pca(df_grouped_scaled)

# Fit clustering model
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(reduced_df)

# Add cluster labels
df_grouped_scaled['cluster'] = clusters

X_columns = ['P_F', 'TA_F', 'WTD_F', 'PA_F', 'G_F', 'H_F', 'NEE_F',
             'bdod_0-5cm_mean', 'cec_0-5cm_mean',
             'phh2o_0-5cm_mean', 'soc_0-5cm_mean', 'nitrogen_0-5cm_mean']

df_cluster_0 = df_grouped_scaled[df_grouped_scaled['cluster'] == 0]
r2_0, mae_0, rmse_0, imp_df0, best_params_0, top_feat_0 = train_and_visualize_rf_with_hyperparam_tuning(df_cluster_0,
                                                                                                        flux_column='FCH4_F',
                                                                                                        X_columns=X_columns)

df_cluster_1 = df_grouped_scaled[df_grouped_scaled['cluster'] == 1]
r2_1, mae_1, rmse_1, imp_df1, best_params_1, top_feat_1 = train_and_visualize_rf_with_hyperparam_tuning(df_cluster_1,
                                                                                                        flux_column='FCH4_F',
                                                                                                        X_columns=X_columns)

# Merge and compare importance dfs
importance_df = pd.concat([imp_df0, imp_df1], axis=1, keys=['Cluster 0', 'Cluster 1'])
importance_df.columns = importance_df.columns.map('_'.join)


# Plot bar chart with feature labels
ax = importance_df.plot(kind='barh', figsize=(12, 8))
ax.set_yticks(range(len(importance_df)))  # Set custom yticks positions

# Set custom tick labels using 'Model1_Feature' column
ax.set_yticklabels(importance_df['Cluster 0_Feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance Comparison', fontsize=12)
plt.legend(fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
plt.savefig('feature_importance_comparison_clusters_sum.png', bbox_inches='tight', dpi=300)