import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([3, 2, 4, 5, 3, 4, 5, 6, 4, 3]).reshape(-1,1)
Y = np.array([5, 3, 6, 2, 4, 5, 3, 2, 4, 6])
Z = np.array([200, 150, 250, 300, 225, 275, 350, 400, 275, 225])

# Criando o modelo de regressão linear
modelo = LinearRegression()
# Treinando o modelo com os dados (horas de estudo e notas)
modelo.fit(X, Y, Z)
# Fazendo previsões usando o modelo treinado
Z_pred = modelo.predict(X)
# Plotando os dados originais e a linha de regressão
plt.scatter(X, Y, Z color='blue') # Dados originais
plt.plot(X, Y, Z_pred color='red') # Linha de regressão
plt.title('Relação entre Horas de Estudo e Notas')
plt.xlabel('Horas de Estudo')
plt.ylabel('Notas')
plt.show()
