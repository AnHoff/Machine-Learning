import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
data = pd.read_csv('14.1 - customer_data.csv')
X = data[['Renda Anual', 'Gastos em Roupas', 'Gastos em Alimentos', 'Gastos em Eletrônicos', 'Gastos em Saúde']]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar GMM
gmm = GaussianMixture(n_components=4, random_state=0)
gmm.fit(X_scaled)
data['Cluster'] = gmm.predict(X_scaled)

# Parâmetros das distribuições gaussianas
means = gmm.means_
covariances = gmm.covariances_

# Visualização dos clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Renda Anual', y='Gastos em Roupas', hue='Cluster', data=data, palette='Set1', s=100, alpha=0.6)
plt.title('Clusters de Clientes (GMM)')
plt.xlabel('Renda Anual')
plt.ylabel('Gastos em Roupas')
plt.legend()
plt.grid(True)
plt.show()
