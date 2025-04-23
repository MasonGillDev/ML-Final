import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, rand_score, mutual_info_score, normalized_mutual_info_score

# Load Titanic data / remove columns with many missing values
df = pd.read_excel('titanic3.xls')
df.drop(['name', 'age', 'cabin', 'boat', 'body', 'home.dest'], axis=1, inplace=True)
df.dropna(inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']

# STEP 1: Standardize the features (mean = 0, std = 1)
# Important because PCA is affected by feature scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 2: Apply PCA to reduce dimensionality
# PCA identifies the directions *PC, that maximize variances
# n_components=2 turns the dataset to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# STEP 3: Apply KMeans clustering on PCA data
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, )
cluster_labels = kmeans.fit_predict(X_pca)

if adjusted_rand_score(y, cluster_labels) < 0:
    cluster_labels = 1 - cluster_labels

# STEP 4: Eval clustering performance
silhouette = silhouette_score(X_pca, cluster_labels)
ch_score = calinski_harabasz_score(X_pca, cluster_labels)
db_score = davies_bouldin_score(X_pca, cluster_labels)
ri = rand_score(y, cluster_labels)
ari = adjusted_rand_score(y, cluster_labels)
mi = mutual_info_score(y, cluster_labels)
nmi = normalized_mutual_info_score(y, cluster_labels)

# Clustering evaluation metrics
print("Clustering Evaluation Metrics:\n")
print(f"Silhouette Score:           {silhouette:.4f}")
print(f"Calinski-Harabasz Score:    {ch_score:.2f}") 
print(f"Davies-Bouldin Index:       {db_score:.4f}")  # Lower is better

# Visualize variance each principal component explains
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', color='b')
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Visualize the KMeans clusters in PCA-space
plt.figure(figsize=(10, 6))
label_names = {0: "Not Survived", 1: "Survived"}
colors = ['red', 'green']

for label in np.unique(cluster_labels):
    plt.scatter(
        X_pca[cluster_labels == label, 0],
        X_pca[cluster_labels == label, 1],
        label=label_names[label],
        alpha=0.6,
        s=60,
        edgecolor='k',
        c=colors[label]
    )

plt.title('KMeans Clustering on PCA Components')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
