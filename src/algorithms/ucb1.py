import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        
    def select_arm(self):
        total_counts = np.sum(self.counts)
        if total_counts < self.n_arms:
            return int(total_counts)  # Elegir cada brazo al menos una vez
        ucb_values = self.values + np.sqrt((2 * np.log(total_counts)) / (self.counts + 1e-5))
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]