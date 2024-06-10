import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Gerar dados de exemplo (altura e peso)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Aplicar GMM
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Visualizar os dados e os clusters
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, s=50, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.grid(True)
plt.show()