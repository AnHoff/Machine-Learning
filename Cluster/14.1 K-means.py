import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Carregar os dados
data = pd.read_csv('14.1 - customer_data.csv')
X = data[['Renda Anual', 'Gastos em Roupas', 'Gastos em Alimentos', 'Gastos em Eletrônicos', 'Gastos em Saúde']]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-means
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_

# Centróides dos clusters
centroids = kmeans.cluster_centers_

# Visualização dos clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Renda Anual', y='Gastos em Roupas', hue='Cluster', data=data, palette='Set1', s=100, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centróides')
plt.title('Clusters de Clientes (K-means)')
plt.xlabel('Renda Anual')
plt.ylabel('Gastos em Roupas')
plt.legend()
plt.grid(True)
plt.show()