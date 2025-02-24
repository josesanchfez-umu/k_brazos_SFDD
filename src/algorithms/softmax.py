import numpy as np
from algorithms.algorithm import Algorithm

class Softmax(Algorithm):
    def __init__(self, k: int, temperature: float):
        """
        Inicializa el algoritmo Softmax para selección de brazos.
        :param k: Número de brazos.
        :param temperature: Parámetro de temperatura para la exploración-explotación.
        """
        super().__init__(k)
        self.temperature = temperature

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax.
        :return: Índice del brazo seleccionado.
        """
        exp_values = np.exp(self.values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(self.k, p=probabilities)

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las recompensas promedio estimadas de cada brazo.
        :param chosen_arm: Índice del brazo que fue seleccionado.
        :param reward: Recompensa obtenida.
        """
        super().update(chosen_arm, reward)

    def reset(self):
        """
        Reinicia el estado del algoritmo y el parámetro de temperatura.
        """
        super().reset()