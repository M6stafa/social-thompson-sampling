import numpy as np

from .policy_estimator import PolicyEstimator


class EWMPolicyEstimator(PolicyEstimator):
    def __init__(self, k: int, ewm_lambda: float = 2/(20+1)) -> None:
        self.ema_lambda = ewm_lambda

        self.n = np.ones(k, dtype=float)

    def update(self, action: int) -> None:
        self.n = (1 - self.ema_lambda) * self.n
        self.n[action] += self.ema_lambda

    def get_policy(self) -> np.ndarray:
        return self.n / np.sum(self.n)
