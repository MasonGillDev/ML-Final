import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, rand_score, mutual_info_score, normalized_mutual_info_score

# Load Titanic dataset
df = pd.read_excel('titanic3.xls')

# Drop features with a lot of missing values
df.drop(['age', 'name', 'cabin', 'boat', 'body', 'home.dest'], axis=1, inplace=True)
df.dropna(inplace=True)  # Drop rows with any missing values

# Encode categorical variables as dummy variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']

# Scale the entire dataset before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the range of PCA components to test
pca_components_list = [2, 5, 10, 50, 100, 500, X.shape[1]]

# Dictionary to store metrics for each PCA configuration
results = {
    'n_components': [],
    'Silhouette': [],
    'CH': [],
    'DBI': [],
    'RI': [],
    'ARI': [],
    'MI': [],
    'NMI': []
}

# Loop over different PCA component settings
for n_components in pca_components_list:
    print(f"Processing PCA with {n_components} components...")

    # Apply PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Perform KMeans clustering - edit clusters
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)

    # Flip cluster labels if they are reversed
    if adjusted_rand_score(y, cluster_labels) < 0:
        cluster_labels = 1 - cluster_labels

    # Compute evaluation metrics
    silhouette = silhouette_score(X_pca, cluster_labels)
    ch_score = calinski_harabasz_score(X_pca, cluster_labels)
    db_score = davies_bouldin_score(X_pca, cluster_labels)
    ri = rand_score(y, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)
    mi = mutual_info_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)

    # Store results
    results['n_components'].append(n_components)
    results['Silhouette'].append(silhouette)
    results['CH'].append(ch_score)
    results['DBI'].append(db_score)
    results['RI'].append(ri)
    results['ARI'].append(ari)
    results['MI'].append(mi)
    results['NMI'].append(nmi)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_excel("titanic_pca_kmeans_results.xlsx", index=False)
print("PCA Results saved to 'titanic_pca_kmeans_results.xlsx'")

# Plotting results
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("plasma")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

metrics = ['Silhouette', 'CH', 'DBI', 'RI', 'ARI', 'MI', 'NMI']
titles = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Index',
          'Rand Index', 'Adjusted Rand Index', 'Mutual Information', 'Normalized Mutual Information']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i]
    ax.plot(results['n_components'], results[metric], marker='o', linewidth=2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('PCA Components', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    for x, y_val in zip(results['n_components'], results[metric]):
        ax.annotate(f'{y_val:.3f}', (x, y_val), textcoords="offset points", xytext=(0, 10), ha='center')
    ax.grid(True)

if len(metrics) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig("titanic_pca_clustering_metrics.png", dpi=300)
print("PCA clustering plots saved to 'titanic_pca_clustering_metrics.png'")

# Normalize metrics for comparison - same as kmeans
normalized_results = results_df.copy()
for metric in metrics:
    min_val = normalized_results[metric].min()
    max_val = normalized_results[metric].max()
    if metric == 'DBI': # lower is better
        normalized_results[metric] = 1 - ((normalized_results[metric] - min_val) / (max_val - min_val))
    else:
        normalized_results[metric] = (normalized_results[metric] - min_val) / (max_val - min_val)

# Print best PCA component count for each metric - same as kmeans just changed for PCA
print("\nBest PCA n_components for each metric:")
for metric in metrics:
    best_idx = results_df[metric].idxmin() if metric == 'DBI' else results_df[metric].idxmax()
    print(f"{metric}: {results_df.loc[best_idx, 'n_components']}")

# Print all results
print("\nPCA Clustering Results:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
