import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Dados de visitantes, publicidade e vendas
visitantes = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
publicidade = np.array([2000, 2500, 3000, 3500, 4000]).reshape(-1, 1)
vendas = np.array([3000, 3500, 4000, 4500, 5000])

# Instanciar e treinar o modelo de regressão linear múltipla
modelo = LinearRegression()
modelo.fit(np.concatenate((visitantes, publicidade), axis=1), vendas)

# Prever as vendas para os dados de treinamento
vendas_preditas = modelo.predict(np.concatenate((visitantes, publicidade), axis=1))

# Plotar os dados originais e a superfície de regressão
fig = plt.figure(figsize=(12, 6))

# Gráfico 3D
ax = fig.add_subplot(121, projection='3d')
ax.scatter(visitantes, publicidade, vendas, c='blue', marker='o')
ax.set_xlabel('Visitantes')
ax.set_ylabel('Publicidade (R$)')
ax.set_zlabel('Vendas (R$)')
ax.set_title('Dados Originais')

# Superfície de regressão
visitantes_grid, publicidade_grid = np.meshgrid(np.linspace(visitantes.min(), visitantes.max(), 100),
                                                np.linspace(publicidade.min(), publicidade.max(), 100))
vendas_grid = modelo.predict(np.array([visitantes_grid.flatten(), publicidade_grid.flatten()]).T).reshape(visitantes_grid.shape)
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(visitantes_grid, publicidade_grid, vendas_grid, alpha=0.5)
ax.scatter(visitantes, publicidade, vendas, c='blue', marker='o')
ax.set_xlabel('Visitantes')
ax.set_ylabel('Publicidade (R$)')
ax.set_zlabel('Vendas (R$)')
ax.set_title('Superfície de Regressão')

plt.tight_layout()
plt.show()

# Prever as vendas para 3500 visitantes e R$ 4.500,00 em publicidade
visitantes_predito = np.array([[3500]])
publicidade_predita = np.array([[4500]])
vendas_preditas = modelo.predict(np.concatenate((visitantes_predito, publicidade_predita), axis=1))

print("Vendas previstas para 3500 visitantes e R$ 4.500,00 em publicidade:", vendas_preditas[0])
