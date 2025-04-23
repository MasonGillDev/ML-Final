import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, rand_score, mutual_info_score, normalized_mutual_info_score

# Load Titanic dataset / drop columns with a lot of missing values
df = pd.read_excel('titanic3.xls')
df.drop(['age', 'cabin', 'boat', 'body', 'home.dest'], axis=1, inplace=True)
df.dropna(inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target variable (y)
X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']

# Define a range of feature selection values (k) to try
# The values represent the number of top features to select for each iteration
k_values = [2, 5, 10, 50, 100, 1000, X.shape[1]]  # Adjusted to Titanic feature count ~2000 but can change depending on fs
results = {
    'k_value': [],
    'Silhouette': [],
    'CH': [],
    'DBI': [],
    'RI': [],
    'ARI': [],
    'MI': [],
    'NMI': []
}

# Loop through different k values and perform KMeans clustering for each subset of features
for k in k_values:
    print(f"Processing k = {k}...")  # Show progress for each k value
    
    # Select the best k features based on ANOVA F-value (using f_classif)
    selector = SelectKBest(f_classif, k=min(k, X.shape[1]))  # Ensure k does not exceed the number of features
    X_selected = selector.fit_transform(X, y)  # Apply the feature selection
    
    # Scale the selected features before clustering - IMPORTANT for KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)  # Standardize the features
    
    # Perform KMeans clustering with 2 clusters (survived vs not survived)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Flip cluster labels if necessary based on the adjusted rand score
    if adjusted_rand_score(y, cluster_labels) < 0:
        cluster_labels = 1 - cluster_labels

    # Compute clustering performance metrics
    silhouette = silhouette_score(X_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    ri = rand_score(y, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)
    mi = mutual_info_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)
    
    # Store the results for the current k value
    results['k_value'].append(k)
    results['Silhouette'].append(silhouette)
    results['CH'].append(ch_score)
    results['DBI'].append(db_score)
    results['RI'].append(ri)
    results['ARI'].append(ari)
    results['MI'].append(mi)
    results['NMI'].append(nmi)

# Create a DataFrame to store the results for easy analysis
results_df = pd.DataFrame(results)
results_df.to_excel("titanic_kmeans_results.xlsx", index=False)
print("Results saved to 'titanic_kmeans_results.xlsx'")

# Plotting the results for each metric with different k values
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # 3x3 grid for plotting
axes = axes.flatten()

# List of metrics to plot
metrics = ['Silhouette', 'CH', 'DBI', 'RI', 'ARI', 'MI', 'NMI']
titles = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Index',
          'Rand Index', 'Adjusted Rand Index', 'Mutual Information', 'Normalized Mutual Information']

# Plot each metric against the k values
for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i]
    ax.plot(results['k_value'], results[metric], marker='o', linewidth=2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Number of Features (k)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    
    # Annotate the points with the metric values
    for x, y_val in zip(results['k_value'], results[metric]):
        ax.annotate(f'{y_val:.3f}', (x, y_val), textcoords="offset points", xytext=(0, 10), ha='center')
    
    ax.grid(True)

if len(metrics) < len(axes):
    fig.delaxes(axes[-1])

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("titanic_clustering_metrics.png", dpi=300)
print("Plots saved to 'titanic_clustering_metrics.png'")

# Normalizing metrics to scale them between 0 and 1 for easier comparison - same code as kmeans
normalized_results = results_df.copy()
for metric in metrics:
    min_val = normalized_results[metric].min()
    max_val = normalized_results[metric].max()
    
    # Normalize DBI (lower is better, so we invert it)
    if metric == 'DBI':
        normalized_results[metric] = 1 - ((normalized_results[metric] - min_val) / (max_val - min_val))
    else:  # Higher is better for other metrics
        normalized_results[metric] = (normalized_results[metric] - min_val) / (max_val - min_val)

# Print the best k value for each metric
print("\nBest k value for each metric:")
for metric in metrics:
    best_idx = results_df[metric].idxmin() if metric == 'DBI' else results_df[metric].idxmax()
    print(f"{metric}: {results_df.loc[best_idx, 'k_value']}")

print("\nFull results:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
