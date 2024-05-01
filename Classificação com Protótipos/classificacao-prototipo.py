import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np

# Carregar os dados do arquivo CSV
dados = pd.read_csv('DataPrototype.csv')

# Vetorizar as frases e as categorias usando TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_frase = tfidf_vectorizer.fit_transform(dados['Frase'])
X_categoria = tfidf_vectorizer.fit_transform(dados['Categoria'])
Y = dados['Categoria']

# Concatenar as features das frases e das categorias
X = np.hstack((X_frase.toarray(), X_categoria.toarray()))

# Codificar as categorias para números
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Dividir os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

# Criar e treinar o modelo NearestCentroid
modelo_prototipos = NearestCentroid()
modelo_prototipos.fit(X_train, Y_train)

# Realizar previsões no conjunto de teste
y_pred = modelo_prototipos.predict(X_test)

# Avaliar o modelo
relatorio = classification_report(Y_test, y_pred)
matriz = confusion_matrix(Y_test, y_pred)

# Exibir resultados
print("Relatório de Classificação:\n", relatorio)
print("Matriz de Confusão:\n", matriz)

# Reduzir a dimensionalidade para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

# Mapear as categorias para cores
cores = {'Tecnologia': 'red', 'Natureza': 'blue', 'Cultura': 'green'}
cores_pred = [cores[label_encoder.classes_[pred]] for pred in y_pred]

# Visualizar os resultados
plt.figure(figsize=(8, 6))
for categoria, cor in cores.items():
    indices = y_pred == label_encoder.transform([categoria])[0]
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], c=cor, label=categoria, edgecolor='black', s=50)
plt.title('Classificação com Protótipos')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()
