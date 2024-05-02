import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import PCA

# Carregar os dados do arquivo CSV
dados = pd.read_csv('DataPrototype.csv')

# Vetorizar as frases usando TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_frase = tfidf_vectorizer.fit_transform(dados['Frase'])
Y = dados['Categoria']

# Criar e treinar o modelo NearestCentroid
modelo_prototipos = NearestCentroid()
modelo_prototipos.fit(X_frase, Y)

# Reduzir a dimensionalidade para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_frase.toarray())

# Mapear as categorias para cores
cores = {'Tecnologia': 'red', 'Natureza': 'blue', 'Cultura': 'green'}
cores_frases = [cores[c] for c in Y]

# Visualizar os resultados
plt.figure(figsize=(8, 6))
for categoria, cor in cores.items():
    indices = Y == categoria
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], c=cor, label=categoria, edgecolor='black', s=50)
plt.title('Frases categorizadas por Protótipos')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
