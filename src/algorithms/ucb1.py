import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):
    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.
        :param k: Número de brazos.
        :param c: Parámetro de ajuste para la exploración (por defecto 1.0).
        """
        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona el brazo con el mayor valor UCB1.
        """
        total_counts = np.sum(self.counts)
        if total_counts < self.k:
            return total_counts  # Explora cada brazo al menos una vez
        
        ucb_values = self.values + self.c * np.sqrt((2 * np.log(total_counts)) / (self.counts + 1e-5))
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las recompensas promedio estimadas de cada brazo.
        """
        super().update(chosen_arm, reward)