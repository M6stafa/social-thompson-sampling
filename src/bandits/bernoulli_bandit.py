import numpy as np

from .bandit import Bandit


class BernoulliBandit(Bandit):
    def __init__(self, ns: list[int], ps: list[np.ndarray]) -> None:
        self.ns = ns
        self.ps = ps

        self.reset()

    def reset(self) -> None:
        rewards = []
        best_actions = []
        last_n = 0
        for max_n, ps in zip(self.ns, self.ps):
            rewards.append(np.random.binomial(
                n=1, p=ps, size=(max_n - last_n, len(ps)),
            ).astype(float))

            best_actions.append(np.full(max_n - last_n, np.argmax(ps), dtype=int))

            last_n = max_n

        self._rewards = np.vstack(rewards)
        self._best_actions = np.hstack(best_actions)

    def act(self, t: int, action: int) -> tuple[float, float]:
        return self._rewards[t-1, action], self._rewards[t-1, self._best_actions[t-1]]
