import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.05, random_state=42)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis", marker="o", edgecolors="k", alpha=0.7
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c="red",
    marker="X",
    label="Centroids",
)

plt.title("K-Means Clustering Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
