import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load dataset (Handwritten digits)
digits = load_digits()
X = digits.data  # Feature matrix (64 features per sample)
# Perform PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Scatter plot of transformed data
plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=digits.target, cmap="viridis", edgecolors="k", alpha=0.7
)
plt.colorbar(label="Digit Label")
plt.title("PCA: Dimensionality Reduction to 2D")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
# Explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_) * 100:.2f}%")
