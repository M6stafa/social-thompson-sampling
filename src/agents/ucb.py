import numpy as np

from .agent import Agent


class UCB(Agent):
    def __init__(self, n: int, k: int) -> None:
        self.k = k

        self.ucb_coeff = 2 * np.log(np.sqrt(n))

        self.reset()

    def reset(self) -> None:
        self.means = np.zeros(self.k, dtype=float)
        self.Ts = np.zeros(self.k, dtype=int)
        self.ucbs = np.full(self.k, np.inf)

    def get_action(self, t: int, *args, **kwargs) -> int:
        return np.argmax(self.ucbs)

    def update(self, t: int, action: int, reward: float, *args, **kwargs) -> None:
        self.means[action] = (self.means[action] * self.Ts[action] + reward) / (self.Ts[action] + 1)
        self.Ts[action] += 1

        self.ucbs[action] = self.means[action] + np.sqrt(self.ucb_coeff / self.Ts[action])
