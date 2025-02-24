import numpy as np
from algorithms.algorithm import Algorithm

class GradientBandit(Algorithm):
    def __init__(self, k: int, alpha: float):
        """
        Inicializa el algoritmo de Gradiente de Preferencias.
        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje para actualizar las preferencias.
        """
        super().__init__(k)
        self.alpha = alpha  # Tasa de aprendizaje
        self.preferences = np.zeros(k)  # H_t(a): Preferencias de cada brazo
        self.avg_reward = 0  # Recompensa promedio móvil
        self.t = 1  # Contador de iteraciones

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la distribución Softmax de las preferencias.
        :return: Índice del brazo seleccionado.
        """
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  # Evitar overflow numérico
        self.probs = exp_prefs / np.sum(exp_prefs)
        return np.random.choice(self.k, p=self.probs)

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las preferencias de los brazos basado en la recompensa recibida.
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        """
        # Actualizar recompensa promedio móvil
        self.avg_reward += (reward - self.avg_reward) / self.t
        self.t += 1  # Incrementar el tiempo

        # Actualizar preferencias de cada brazo
        for a in range(self.k):
            if a == chosen_arm:
                self.preferences[a] += self.alpha * (reward - self.avg_reward) * (1 - self.probs[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.avg_reward) * self.probs[a]

    def reset(self):
        """Reinicia las preferencias y las recompensas promedio."""
        super().reset()
        self.preferences = np.zeros(self.k)
        self.avg_reward = 0
        self.t = 1
