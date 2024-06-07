import numpy as np

from .agent import Agent


class ThompsonSamplingAgent(Agent):
    def __init__(self, k: int, history_length: int|None = None) -> None:
        self.k = k
        self.history_length = history_length

        self.reset()

    def reset(self) -> None:
        self._rewards_history = [[] for _ in range(self.k)]

        self._beta_dists_log = []
        self._save_beta_dists()

    def get_action(self, t: int, *args, **kwargs) -> int:
        return np.argmax([np.random.beta(a, b) for a, b in self.get_beta_dists()])

    def update(self, t: int, action: int, reward: float, *args, **kwargs) -> None:
        self._rewards_history[action].append(int(reward > 0.5))

        self._save_beta_dists()

    def get_logs(self):
        return {'action_beta_dists': np.array(self._beta_dists_log)}

    def get_beta_dists(self) -> np.ndarray:
        # return the parameters of beta dist for each action in form of (alpha, beta)
        beta_dists = np.empty((self.k, 2), dtype=int)
        start_index = 0 if self.history_length is None else -self.history_length
        for action, action_history in enumerate(self._rewards_history):
            beta_dists[action] = np.bincount(1 - np.array(action_history[start_index:], dtype=int), minlength=2) + 1

        return beta_dists

    def _save_beta_dists(self) -> None:
        self._beta_dists_log.append(np.copy(self.get_beta_dists()))
