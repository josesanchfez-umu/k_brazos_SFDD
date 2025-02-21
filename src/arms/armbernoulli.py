import numpy as np
from arms import Arm

class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución Bernoulli.

        :param p: Probabilidad de éxito (de obtener recompensa 1).
        """
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0,1]."
        self.p = p

    def pull(self):
        """
        Genera una recompensa de 0 o 1 con probabilidad p.
        """
        return np.random.binomial(1, self.p)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.
        """
        return self.p

    def __str__(self):
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int):
        """
        Genera k brazos Bernoulli con probabilidades aleatorias entre 0.1 y 0.9.
        """
        probabilities = np.random.uniform(0.1, 0.9, size=k)
        return [ArmBernoulli(p) for p in probabilities]
