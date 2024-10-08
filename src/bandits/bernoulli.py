import numpy as np

from .base import BaseBandit


class BernoulliBandit(BaseBandit):
    def __init__(self, n: int, ps: np.ndarray) -> None:
        self.n = n
        self.ps = ps

    def reset(self) -> None:
        self._rewards = np.random.binomial(
            n=1, p=self.ps, size=(self.n, len(self.ps)),
        ).astype(float)

        self._best_action = np.argmax(self.ps)

    def act(self, t: int, action: int) -> float:
        return self._rewards[t-1, action]

    def best_action_reward(self, t: int) -> tuple[int, float]:
        return self._best_action, self._rewards[t-1, self._best_action]
