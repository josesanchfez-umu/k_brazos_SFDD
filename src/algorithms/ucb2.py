import numpy as np

from algorithms.algorithm import Algorithm

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2.
        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste para la exploración-explotación (0 < alpha < 1).
        """
        super().__init__(k)
        self.alpha = alpha
        self.epochs = np.zeros(k, dtype=int)
        self.tau = np.ones(k)  # Aseguramos que tau nunca sea menor que 1.

    def select_arm(self) -> int:
        """
        Selecciona el brazo con el mayor valor UCB2.
        """
        total_counts = np.sum(self.counts)
        if total_counts < self.k:
            return total_counts  # Explora cada brazo al menos una vez

        # Evitar división por cero y valores negativos en logaritmo
        safe_tau = np.maximum(self.tau, 1)
        safe_counts = np.maximum(total_counts, 1)

        ucb_values = self.values + np.sqrt(((1 + self.alpha) * np.log(np.e * safe_counts / safe_tau)) / (2 * safe_tau))

        return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las recompensas promedio estimadas de cada brazo.
        """
        super().update(chosen_arm, reward)

        # Incrementar la época y asegurar que tau no crezca exponencialmente
        self.epochs[chosen_arm] += 1
        self.tau[chosen_arm] = np.minimum(np.ceil((1 + self.alpha) ** self.epochs[chosen_arm]), 1e6)  # Límite superior
