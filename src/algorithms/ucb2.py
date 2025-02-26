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
        
        ucb_values = self.values + np.sqrt(((1 + self.alpha) * np.log(np.e * total_counts / self.tau)) / (2 * self.tau))
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las recompensas promedio estimadas de cada brazo.
        """
        super().update(chosen_arm, reward)
        
        # Actualización de la época
        self.epochs[chosen_arm] += 1
        self.tau[chosen_arm] = np.ceil((1 + self.alpha) ** self.epochs[chosen_arm])