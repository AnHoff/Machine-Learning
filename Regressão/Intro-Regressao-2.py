import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28]).reshape(-1,1)
Y = np.array([100, 120, 140, 160, 180, 200, 220, 240, 260, 280])

# Criando o modelo de regressão linear
modelo = LinearRegression()
# Treinando o modelo com os dados (horas de estudo e notas)
modelo.fit(X, Y)
# Fazendo previsões usando o modelo treinado
Y_pred = modelo.predict(X)
# Plotando os dados originais e a linha de regressão
plt.scatter(X, Y, color='blue') # Dados originais
plt.plot(X, Y_pred, color='red') # Linha de regressão
plt.title('Relação entre Temperatura Média (ºC) e Bicicletas Alugadas')
plt.xlabel('Temperatura Média')
plt.ylabel('Bicicletas Alugadas')
plt.show()
