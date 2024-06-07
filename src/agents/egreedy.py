from collections.abc import Callable

import numpy as np

from .agent import Agent


class EGreedy(Agent):
    def __init__(self, epsilon: Callable[[float], float], k: int) -> None:
        self.epsilon = epsilon
        self.k = k

        self.reset()

    def reset(self) -> None:
        self.means = np.zeros(self.k, dtype=float)
        self.Ts = np.zeros(self.k, dtype=int)

    def get_action(self, t: int, *args, **kwargs) -> int:
        if np.random.uniform() <= self.epsilon(t):
            return np.random.randint(self.k)
        return np.argmax(self.means)

    def update(self, t: int, action: int, reward: float, *args, **kwargs) -> None:
        self.means[action] = (self.means[action] * self.Ts[action] + reward) / (self.Ts[action] + 1)
        self.Ts[action] += 1
