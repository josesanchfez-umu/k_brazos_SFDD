import numpy as np
from arms import Arm

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución Binomial.

        :param n: Número de intentos en cada jugada.
        :param p: Probabilidad de éxito en cada intento.
        """
        assert 0 < n, "El número de intentos n debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0,1]."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Binomial B(n, p).
        """
        return np.random.binomial(self.n, self.p)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Binomial.
        """
        return self.n * self.p

    def __str__(self):
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n_min: int = 5, n_max: int = 20):
        """
        Genera k brazos binomiales con parámetros aleatorios.
        """
        n_values = np.random.randint(n_min, n_max + 1, size=k)
        p_values = np.random.uniform(0.1, 0.9, size=k)

        return [ArmBinomial(n, p) for n, p in zip(n_values, p_values)]
