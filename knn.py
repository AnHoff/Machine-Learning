import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
iris_data = pd.read_csv('9-1-knn.csv')

# Separar as variáveis independentes e a variável dependente
X = iris_data.iloc[:, :-1].values
y = iris_data.iloc[:, -1].values

# Dividir os dados em conjuntos de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo K-NN
modelo_knn = KNeighborsClassifier(n_neighbors=3)
modelo_knn.fit(X_train, y_train)

# Realizar previsões com o modelo nos dados de teste
y_pred = modelo_knn.predict(X_test)

# Avaliar o modelo
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))

# Gráficos de dispersão
plt.figure(figsize=(15, 10))

# Gráfico 1: Idade vs. Tempo de Contrato
plt.subplot(231)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='o')
plt.xlabel('Tempo de Contrato')
plt.ylabel('Idade')
plt.title('Idade vs. Tempo de Contrato')
plt.colorbar(label='Churn')

# Gráfico 2: Idade vs. Gastos Mensais
plt.subplot(232)
plt.scatter(X_test[:, 2], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='o')
plt.xlabel('Gastos Mensais')
plt.ylabel('Idade')
plt.title('Idade vs. Gastos Mensais')
plt.colorbar(label='Churn')

# Gráficos 3D

# Preparar os dados
tempo_contrato = iris_data['TempoContrato']
gastos_mensais = iris_data['GastosMensais']
troca_plano = iris_data['TrocaPlano']
idade = iris_data['Idade']

# Gráfico 3D-1:
# Criar figura e eixos 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotar o gráfico tridimensional
ax.scatter(tempo_contrato, gastos_mensais, troca_plano, c=iris_data['Churn'], cmap='coolwarm', s=50)

# Configurar rótulos dos eixos e título
ax.set_xlabel('Tempo de Contrato')
ax.set_ylabel('Gastos Mensais')
ax.set_zlabel('Troca de Plano')
ax.set_title('Gráfico Tridimensional: Tempo de Contrato, Gastos Mensais e Troca de Plano')

# Gráfico 3D-2:
# Criar figura e eixos 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotar o gráfico tridimensional
ax.scatter(idade, tempo_contrato, gastos_mensais, c=iris_data['Churn'], cmap='coolwarm', s=50)
ax.set_xlabel('Idade')
ax.set_ylabel('Tempo de Contrato')
ax.set_zlabel('Gastos Mensais')
ax.set_title('Gráfico Tridimensional: Idade, Tempo de Contrato e Gastos Mensais')

# Exibir todos os gráficos
plt.tight_layout()
plt.show()
