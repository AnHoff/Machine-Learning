import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Passo 1: Carregar os dados do arquivo CSV
dados = pd.read_csv('DataLeastSquare2.csv')

# Visualizar as primeiras linhas dos dados
print(dados.head())

# Passo 2: Ajustar um modelo de regressão linear múltipla
# Separar as variáveis independentes (X) e a variável dependente (y)
X = dados[['Temperatura', 'VelocidadeVento', 'Precipitacao']]
y = dados['BicicletasAlugadas']

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão linear múltipla
modelo = LinearRegression()

# Treinar o modelo com os dados de treinamento
modelo.fit(X_train, y_train)

# Passo 3: Realizar uma análise de diagnóstico
# Coeficientes do modelo
print("Coeficientes:")
for coef, feature in zip(modelo.coef_, X.columns):
    print(feature, ':', coef)
print("Intercepto:", modelo.intercept_)

# Avaliar o modelo nos dados de teste
y_pred = modelo.predict(X_test)
print("Erro quadrático médio (MSE):", mean_squared_error(y_test, y_pred))
print("Coeficiente de determinação (R²):", r2_score(y_test, y_pred))

# Passo 4: Visualizar a relação entre as variáveis independentes e a variável dependente
# Gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test['Temperatura'], X_test['VelocidadeVento'], X_test['Precipitacao'], c=y_test, cmap='viridis', label='Dados de teste')
ax.set_xlabel('Temperatura')
ax.set_ylabel('Velocidade do Vento')
ax.set_zlabel('Precipitação')
ax.set_title('Relação entre Variáveis Independentes e Bicicletas Alugadas')
plt.colorbar(scatter, label='Bicicletas Alugadas')
plt.legend()
plt.show()

# Passo 5: Interpretar os coeficientes
# A interpretação dos coeficientes depende da escala das variáveis, mas em geral:
# - Um coeficiente positivo indica que um aumento na variável independente está associado a um aumento na variável dependente.
# - Um coeficiente negativo indica que um aumento na variável independente está associado a uma diminuição na variável dependente.

# Passo 6: Utilizar o modelo para fazer previsões
# Podemos utilizar o modelo para prever o número de bicicletas alugadas para novos dados.

# Exemplo de previsão para novos dados
novos_dados = [[25, 20, 5]]  # Temperatura, velocidade do vento e precipitação
previsao = modelo.predict(novos_dados)
print("Previsão para os novos dados:", previsao[0])