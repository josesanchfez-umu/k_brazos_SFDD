"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy, Softmax


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, Softmax):
        label += f" (temperature={algo.temperature})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Porcentaje de selección óptima', fontsize=14)
    plt.title('Porcentaje de selección óptima vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


# def plot_arm_statistics(arm_stats: LoQueConsideres, algorithms: List[Algorithm], *args):
def plot_arm_statistics(arm_stats: List, algorithms: List[Algorithm], optimal_arm: int):
    """
    Genera gráficas separadas de Selección de Arms: Ganancias vs Pérdidas para cada algoritmo.

    :param arm_stats: Diccionario con estadísticas de cada brazo por algoritmo.
                      { "algoritmo1": [(selecciones, recompensa_promedio), ...],
                        "algoritmo2": [(selecciones, recompensa_promedio), ...] }
    :param algorithms: Lista de nombres de algoritmos comparados.
    :param optimal_arm: Índice del brazo óptimo.
    """
    
    fig, axes = plt.subplots(1, len(algorithms), figsize=(25, 5), sharey=True)

    if len(algorithms) == 1:
        axes = [axes]  # Para evitar errores si hay un solo algoritmo
    
    for ax, (algo, stats) in zip(axes, arm_stats):
        arms = np.arange(len(stats))  # Índices de los brazos
        selections = [s[0] for s in stats.values()]
        rewards = [s[1] for s in stats.values()]
        
        bars = ax.bar(arms, rewards, color=['#FF9999' if i == optimal_arm else '#99B3FF' for i in arms])
        ax.set_title(f"{get_algorithm_label(algo)}")
        ax.set_xlabel("Brazos")
        ax.set_ylabel("Recompensa Promedio")

        # Agregar etiquetas con el número de selecciones en cada barra
        for bar, num in zip(bars, selections):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{num}", ha='center', va='bottom')

    plt.suptitle("Estadísticas de selección de brazos")
    plt.show()


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
    """

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    
    for idx, algo in enumerate(algorithms):
        # Función que devuelve el nombre del algoritmo
        label = get_algorithm_label(algo) 
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    # Agregar cota teórica de regret (opcional)
    if args:
        cte_ln_t = args[0]
        # Cota teórica C * ln(T)
        theoretical_bound = cte_ln_t * np.log(np.arange(1, steps + 1))
        plt.plot(range(steps), theoretical_bound, 'k--', label='Cota Teórica C * ln(T)', linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title('Regret Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()
