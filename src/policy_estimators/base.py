from abc import ABC, abstractmethod

import numpy as np


class BasePolicyEstimator(ABC):
    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms

        self.reset()

    def reset(self) -> None:
        # Start with uniform policy
        self._policy = np.full(self.n_arms, 1 / self.n_arms, dtype=float)

        self._logs: list[np.ndarray] = []
        self.save_log()

    @abstractmethod
    def update(self, action: int) -> None:
        # Update self._policy in this function
        raise NotImplementedError()

    def get_policy(self) -> np.ndarray:
        return self._policy

    def save_log(self) -> None:
        self._logs.append(self._policy.copy())

    def get_logs(self) -> list[np.ndarray]:
        return self._logs
