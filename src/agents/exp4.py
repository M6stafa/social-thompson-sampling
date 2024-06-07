import random

import numpy as np

from .agent import Agent


class EXP4(Agent):
    def __init__(self, n: int, k: int, n_experts: int, gamma: float = 1e-6) -> None:
        self.n = n
        self.k = k
        self.n_experts = n_experts
        self.gamma = gamma

        self.eta = np.sqrt(2 * np.log(self.n_experts) / (self.n * self.k))

        self.reset()

    def reset(self) -> None:
        self.Q = np.empty((self.n + 1, self.n_experts))
        self.Q[0] = 1 / self.n_experts

    def get_action(self, t: int, experts: np.ndarray, *args, **kwargs) -> int:
        self.Pt = self.Q[t-1] @ experts
        return random.choices(range(self.k), weights=self.Pt)[0]

    def update(self, t: int, action: int, reward: float, experts: np.ndarray, *args, **kwargs) -> None:
        X_hat = np.full(self.k, 1)
        X_hat[action] -= (1 - reward) / (self.Pt[action] + self.gamma)

        X_tilde = experts @ X_hat

        self.Q[t] = np.exp(self.eta * X_tilde) * self.Q[t-1]
        self.Q[t] /= np.sum(self.Q[t])
