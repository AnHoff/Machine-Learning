import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de carga de CPU e tempo de resposta
carga_cpu = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)  # reshape para compatibilidade com sklearn
tempo_resposta = np.array([40, 50, 60, 70, 80])

# Instanciar e treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(carga_cpu, tempo_resposta)

# Coeficiente (inclinação) e intercepto da linha de melhor ajuste
inclinacao = modelo.coef_[0]
intercepto = modelo.intercept_

# Prever o tempo de resposta para os dados de carga de CPU
tempo_resposta_predito = modelo.predict(carga_cpu)

# Carga de CPU para previsão
carga_cpu_predita = np.array([[60]])  # reshape para compatibilidade com sklearn
tempo_resposta_predito_60 = modelo.predict(carga_cpu_predita)

# Plotar os dados originais e a linha de regressão linear
plt.figure(figsize=(8, 6))
plt.scatter(carga_cpu, tempo_resposta, color='blue', label='Dados Originais')
plt.plot(carga_cpu, tempo_resposta_predito, color='red', label='Linha de Regressão Linear')
plt.scatter(carga_cpu_predita, tempo_resposta_predito_60, color='green', label='Carga de CPU 60% (Previsto)')
plt.title('Carga de CPU vs Tempo de Resposta')
plt.xlabel('Carga de CPU (%)')
plt.ylabel('Tempo de Resposta (ms)')
plt.legend()
plt.grid(True)
plt.show()
