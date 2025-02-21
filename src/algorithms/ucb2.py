import numpy as np

from algorithms.algorithm import Algorithm

class UCB2(Algorithm):
    def __init__(self, n_arms, alpha=0.1):
        self.n_arms = n_arms
        self.alpha = alpha
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.epochs = np.zeros(n_arms, dtype=int)
        self.tau = np.ones(n_arms)
    
    def select_arm(self):
        total_counts = np.sum(self.counts)
        if total_counts < self.n_arms:
            return int(total_counts)  # Elegir cada brazo al menos una vez
        
        ucb_values = self.values + np.sqrt(((1 + self.alpha) * np.log(np.e * total_counts / self.tau)) / (2 * self.tau))
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]
        
        # Actualización de la época
        self.epochs[chosen_arm] += 1
        self.tau[chosen_arm] = np.ceil((1 + self.alpha) ** self.epochs[chosen_arm])