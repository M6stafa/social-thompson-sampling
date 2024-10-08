import numpy as np

from .base import BasePolicyEstimator


class EWMPolicyEstimator(BasePolicyEstimator):
    def __init__(self, n_arms: int, ewm_lambda: float = 2 / (20 + 1)) -> None:
        super().__init__(n_arms)

        self.ewm_lambda = ewm_lambda

        self.reset()

    def reset(self) -> None:
        super().reset()

        self.n = np.ones(self.n_arms, dtype=float)

    def update(self, action: int) -> None:
        self.n = (1 - self.ewm_lambda) * self.n
        self.n[action] += self.ewm_lambda

        self._policy = self.n / np.sum(self.n)
