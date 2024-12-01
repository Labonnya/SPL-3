import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Generate or load data (using a random dataset here for demonstration)
# You can replace this with your dataset loaded using pd.read_csv('your_dataset.csv')
np.random.seed(42)
data = np.random.rand(100, 2)  # 100 samples with 2 features
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2'])

# Step 2: Preprocess data (scaling)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Apply KMeans Clustering
# Choose the number of clusters (K), here we assume 3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Step 4: Add cluster labels to the original data
df['Cluster'] = kmeans.labels_

# Step 5: Visualize the clusters
plt.scatter(df['Feature_1'], df['Feature_2'], c=df['Cluster'], cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Step 6: Print cluster centers and labels
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Labels:\n", df['Cluster'].value_counts())
