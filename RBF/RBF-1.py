import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados do arquivo CSV
dados = pd.read_csv('DataRBF1.csv')

# Separar os dados em variáveis independentes e dependentes
X = dados[['Temperatura']].values
y = dados['ConsumoEnergia'].values

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo SVR com kernel RBF
model_rbf = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
model_rbf.fit(X_train, y_train)

# Realizar previsões com o modelo
y_pred = model_rbf.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMétricas de desempenho:")
print("Erro médio quadrático (MSE):", mse)
print("Coeficiente de determinação (R²):", r2)

# Visualização dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Dados de Treino')
plt.scatter(X_test, y_test, color='green', label='Dados de Teste')
plt.plot(X_test, y_pred, color='red', label='Modelo RBF')
plt.title('Consumo de Energia em relação à Temperatura (SVR com kernel RBF)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Consumo de Energia')
plt.legend()
plt.show()
