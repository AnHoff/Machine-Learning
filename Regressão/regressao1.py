import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1,1)
Y = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

# Criando o modelo de regressão linear
modelo = LinearRegression()

# Treinando o modelo com os dados (horas de estudo e notas)
modelo.fit(X, Y)

# Fazendo previsões usando o modelo treinado
Y_pred = modelo.predict(X)

# Plotando os dados originais e a linha de regressão
plt.scatter(X, Y, color='blue') # Dados originais
plt.plot(X, Y_pred, color='red') # Linha de regressão
plt.title('Relação entre Horas de Estudo e Notas')
plt.xlabel('Horas de Estudo')
plt.ylabel('Notas')
plt.show()