import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parámetros de las distribuciones
mu1 = 0  # media de la distribución p(x)
mu2 = 2  # media de la distribución q(x)
sigma = 1  # desviación estándar (igual para ambas distribuciones

# Crear el rango de valores x
x = np.linspace(mu1 - 4*sigma, mu2 + 4*sigma, 1000)

# Calcular las densidades de probabilidad
p_x = norm.pdf(x, mu1, sigma)
q_x = norm.pdf(x, mu2, sigma)

# Calcular la divergencia KL utilizando la fórmula derivada
kl_div = p_x * np.log(p_x / q_x)

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(x, p_x, label='$p(x) = N(\mu_1, \sigma)$', color='blue')
plt.plot(x, q_x, label='$q(x) = N(\mu_2, \sigma)$', color='red')
plt.fill_between(x, kl_div, color='green', alpha=0.5, label='$D_{KL}(p \| q)$')

# Añadir títulos y etiquetas
plt.title('Divergencia de Kullback-Leibler entre $N(\mu_1, \sigma)$ y $N(\mu_2, \sigma)$')
plt.xlabel('x')
plt.ylabel('Densidad de probabilidad')
plt.legend()

# Mostrar la gráfica
plt.grid(True)
plt.show()
