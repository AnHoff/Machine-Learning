import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados do arquivo CSV
caminho_csv = 'DataRBF2.csv'
dados = pd.read_csv(caminho_csv)

# Explorar os dados
print("Primeiras linhas dos dados:")
print(dados.head())
print("\nDescrição estatística dos dados:")
print(dados.describe())

# Separar os dados em variáveis independentes e dependentes
X = dados[['Temperatura', 'VelocidadeVento', 'Precipitacao']].values
y = dados['BicicletasAlugadas'].values

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir e treinar o modelo SVR com kernel RBF
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
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.title('Desempenho do Modelo SVR com Kernel RBF')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.show()