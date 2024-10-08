from typing import Any

import numpy as np
from scipy import stats

from .base import BaseAgent
from ..worlds.agent_data import AgentData


class ThompsonSamplingAgent(BaseAgent):
    def __init__(self, n_arms: int, history_length: int|None = None) -> None:
        self.n_arms = n_arms
        self.history_length = history_length

        self.reset()

    def reset(self) -> None:
        self._rewards_history = [[] for _ in range(self.n_arms)]

        self._beta_dists_log = []
        self._beta_dists = self.get_beta_dists()

    def get_action(self, t: int, neighbor_agents: dict[str, AgentData]) -> int:
        return np.argmax([np.random.beta(a, b) for a, b in self._beta_dists])

    def update(self, t: int, action: int, reward: float, neighbor_agents: dict[str, AgentData]) -> None:
        self._save_beta_dists(self._beta_dists)

        self._rewards_history[action].append(int(reward > 0.5))
        self._beta_dists = self.get_beta_dists()

    def get_beta_dists(self) -> np.ndarray:
        # return the parameters of beta dist for each action in form of (alpha, beta)
        beta_dists = np.empty((self.n_arms, 2), dtype=int)
        start_index = 0 if self.history_length is None else -self.history_length
        for action, action_history in enumerate(self._rewards_history):
            beta_dists[action] = np.bincount(1 - np.array(action_history[start_index:], dtype=int), minlength=2) + 1

        return beta_dists

    def get_policy(self, n_samples: int = 500) -> np.ndarray:
        samples = []
        for a, b in self._beta_dists:
            samples.append(stats.beta(a, b).rvs(n_samples))

        selection_probs = np.bincount(np.argmax(samples, axis=0), minlength=len(self._beta_dists))
        selection_probs = selection_probs / n_samples

        return selection_probs

    def _save_beta_dists(self, beta_dists: np.ndarray|None = None) -> None:
        self._beta_dists_log.append(np.copy(self.get_beta_dists() if beta_dists is None else beta_dists))

    def get_logs(self) -> dict[str, Any]|None:
        return {'action_beta_dists': np.array(self._beta_dists_log)}
