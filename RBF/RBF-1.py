import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados do arquivo CSV
dados = pd.read_csv('DataRBF1.csv')

# Separar os dados em variáveis independentes e dependentes
X = dados[['Temperatura']].values
y = dados['ConsumoEnergia'].values

# Treinar um modelo SVR com kernel RBF
model_rbf = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
model_rbf.fit(X, y)

# Realizar previsões com o modelo para todos os dados de temperatura
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model_rbf.predict(X_range)

# Avaliar o modelo
mse = mean_squared_error(y, model_rbf.predict(X))
r2 = r2_score(y, model_rbf.predict(X))
print("\nMétricas de desempenho:")
print("Erro médio quadrático (MSE):", mse)
print("Coeficiente de determinação (R²):", r2)

# Visualização dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados')
plt.plot(X_range, y_pred, color='red', label='Modelo RBF')
plt.title('Consumo de Energia em relação à Temperatura (SVR com kernel RBF)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Consumo de Energia')
plt.legend()
plt.show()
