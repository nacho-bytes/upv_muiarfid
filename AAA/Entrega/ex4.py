import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class RegularizedGaussianMixtureEM:
    def __init__(self, n_components, max_iter=100, tol=1e-4, gamma=0.1, init_means=None, init_var=None, init_weights=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.weights = init_weights
        self.means = init_means
        self.variances = init_var

    def fit(self, data):

        # Inicialización de parámetros
        np.random.seed(42)
        if self.means is None:
            self.means = np.random.choice(data, self.n_components, replace=False)
        if self.variances is None:
            self.variances = np.random.random(self.n_components)
        if self.weights is None:
            self.weights = np.ones(self.n_components) / self.n_components
        # Almacenar valores de pi_k en cada iteración
        self.pi_history = [self.weights.copy()]
        self.means_history = [self.means.copy()]
        self.variances_history = [self.variances.copy()]

        log_likelihood = 0
        for _ in range(self.max_iter):
            # Paso E
            responsibilities = self.e_step(data)

            # Paso M
            self.m_step(data, responsibilities)

            # Almacenar valores de pi_k
            self.pi_history.append(self.weights.copy())
            self.means_history.append(self.means.copy())
            self.variances_history.append(self.variances.copy())

            # Calcular log-verosimilitud
            new_log_likelihood = self.compute_log_likelihood(data)
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

    def e_step(self, data):
        weighted_log_prob = np.array([w * norm.pdf(data, mean, np.sqrt(var))
                                      for w, mean, var in zip(self.weights, self.means, self.variances)])
        responsibilities = weighted_log_prob / weighted_log_prob.sum(axis=0)
        return responsibilities.T
    
    def m_step(self, data, responsabilities):
        log_responsabilities = np.log(responsabilities + 1e-10)
        weights_numerador = responsabilities * (1 + self.gamma * log_responsabilities)
        self.weights = np.sum(weights_numerador, axis=0) / np.sum(weights_numerador) #Actualizamos pesos con REGULARIZACIÓN
        aux1 = data[:, np.newaxis] * weights_numerador
        self.means = np.sum(aux1, axis=0) / np.sum(weights_numerador, axis=0) 
    
    def compute_log_likelihood(self, data):
        log_prob = np.log([w * norm.pdf(data, mean, np.sqrt(var)) + 1e-10
                           for w, mean, var in zip(self.weights, self.means, self.variances)])
        return log_prob.sum()

# Generar datos de ejemplo de una mezcla de gaussianas
np.random.seed(42)
n_samples = 1000
n_features = 1

means_og = np.array([-6, 2])
variances_og = np.array([4.0, 4.0])
weights_og = np.array([0.2, 0.8])

data = []
for mean, cov, weight in zip(means_og, variances_og, weights_og):
    data.append(np.random.normal(mean, np.sqrt(cov), int(weight * n_samples)))
data = np.hstack(data)
np.random.shuffle(data)

# Entrenar el modelo
gmm_em = RegularizedGaussianMixtureEM(
    n_components=3, 
    gamma=0.0, 
    max_iter=26, 
    init_var=np.array([4, 4, 4]), 
    init_means=np.array([-2, 4, 5]),
    init_weights=np.array([0.5, 0.3, 0.2])
)
gmm_em.fit(data)

# Print final parameters
print("Medias:", gmm_em.means)
print("Pesos:", gmm_em.weights)
print("Varianzas:", gmm_em.variances)


pi_history = np.array(gmm_em.pi_history)
means_history = np.array(gmm_em.means_history)
variances_history = np.array(gmm_em.variances_history)
iterations = np.arange(pi_history.shape[0])

fig, axs = plt.subplots(3, 3, figsize=(12, 10))
fig.suptitle(f'Regularization: $\gamma = {gmm_em.gamma}$')
axs = axs.flatten()


selected_iterations = [0, 1, 2, 5, 10, 15, 18, 20, 26]

x = np.linspace(data.min(), data.max(), 1000)
for i, iter_idx in enumerate(selected_iterations, start=-1):

    means = means_history[iter_idx]
    variances = variances_history[iter_idx]
    weights = pi_history[iter_idx]


    for k in range(gmm_em.n_components):
        axs[i+1].plot(x, weights[k] * norm.pdf(x, means[k], np.sqrt(variances[k])), linestyle='--', label=f'G_n{k + 1}')

 
    combined_pdf = np.sum([weights[k] * norm.pdf(x, means[k], np.sqrt(variances[k])) for k in range(gmm_em.n_components)], axis=0)
    real_pdf = np.sum([w * norm.pdf(x, m, np.sqrt(variance)) for w, m, variance in zip(weights_og, means_og, variances_og)], axis=0)
    axs[i+1].plot(x, real_pdf, color='grey', label="G mixture")
    axs[i+1].plot(x, combined_pdf, color='black', linestyle='--', label='Predicted G mixture')

    axs[i+1].hist(data, bins=60, density=True, alpha=0.6, color='red')

    # Usar LaTeX para el título
    weights_str = ", ".join([f"$\\pi_{{{idx}}} = {w:.3f}$" for idx, w in enumerate(weights)])
    means_str = ", ".join([f"$\\mu_{{{idx}}} = {m:.3f}$" for idx, m in enumerate(means)])
    axs[i+1].set_title(f'$iter:{{{iter_idx}}}$ - {weights_str} \n- {means_str}')
    # axs[i+1].grid(True)
    axs[i+1].legend()
    
plt.tight_layout()
plt.show()