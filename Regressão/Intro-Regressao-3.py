import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Dados
X = np.array([[3, 5],
              [2, 3],
              [4, 6],
              [5, 2],
              [3, 4],
              [4, 5],
              [5, 3],
              [6, 2],
              [4, 4],
              [3, 6]])  # Anúncios em redes sociais e jornais
Z = np.array([200, 150, 250, 300, 225, 275, 350, 400, 275, 225])  # Vendas do dia (em milhares)

# Criando o modelo de regressão linear
modelo = LinearRegression()
# Treinando o modelo com os dados
modelo.fit(X, Z)
# Fazendo previsões usando o modelo treinado
Z_pred = modelo.predict(X)

# Coeficientes da regressão
coeficientes = modelo.coef_
intercept = modelo.intercept_
print("Coeficientes:", coeficientes)
print("Intercept:", intercept)

# Plotando os dados originais e a linha de regressão
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Z, c='b', marker='o')

# Adicionando a linha de regressão
x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
Z_surf = modelo.intercept_ + modelo.coef_[0] * x_surf + modelo.coef_[1] * y_surf
ax.plot_surface(x_surf, y_surf, Z_surf, color='red', alpha=0.5)

ax.set_xlabel('Anúncios em Redes Sociais')
ax.set_ylabel('Anúncios em Jornais')
ax.set_zlabel('Vendas do dia (milhares)')

plt.title('Previsão de Vendas com Base em Anúncios em Jornais e Redes Sociais')
plt.show()
