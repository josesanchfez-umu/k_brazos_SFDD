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
        self.tau = np.ones(k)

    def select_arm(self) -> int:
        """
        Selecciona el brazo con el mayor valor UCB2.
        """
        total_counts = np.sum(self.counts)
        if total_counts < self.k:
            return total_counts  # Explora cada brazo al menos una vez

        safe_tau = np.maximum(self.tau, 1)
        safe_counts = np.maximum(total_counts, 1)
        safe_log_term = np.maximum(np.e * safe_counts / safe_tau, 1.0001)  # Evita valores ≤ 0 en log()

        ucb_values = self.values + np.sqrt(((1 + self.alpha) * np.log(safe_log_term)) / (2 * safe_tau))

        return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las recompensas promedio estimadas de cada brazo.
        """
        super().update(chosen_arm, reward)

        # Control del crecimiento exponencial para evitar overflow
        max_exp = 100  # Evita crecimiento excesivo
        self.epochs[chosen_arm] = np.minimum(self.epochs[chosen_arm] + 1, max_exp)

        self.tau[chosen_arm] = np.minimum(np.ceil((1 + self.alpha) ** self.epochs[chosen_arm]), 1e6)
