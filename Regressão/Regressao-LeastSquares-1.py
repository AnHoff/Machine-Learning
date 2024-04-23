import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Carregando os dados do arquivo CSV
dados = pd.read_csv('DataLeastSquare1.csv')

# Visualização dos dados
plt.scatter(dados['Temperatura_Media'], dados['Bicicletas_Alugadas'], color='blue', label='Dados')
plt.xlabel('Temperatura Média Diária (°C)')
plt.ylabel('Número de Bicicletas Alugadas')
plt.title('Relação entre Temperatura Média e Aluguel de Bicicletas')
plt.legend()

# Preparação dos dados para a regressão
X = dados['Temperatura_Media'].values.reshape(-1, 1)  # Temperatura Média como feature
y = dados['Bicicletas_Alugadas'].values

# Criação do modelo de regressão linear
modelo = LinearRegression()

# Treinamento do modelo
modelo.fit(X, y)

# Coeficientes da regressão
intercept = modelo.intercept_
slope = modelo.coef_[0]

# Plotando a linha de regressão
plt.plot(X, modelo.predict(X), color='red', label='Linha de Regressão')
plt.legend()

plt.show()

# Saída dos coeficientes
print("Intercepto:", intercept)
print("Inclinação:", slope)