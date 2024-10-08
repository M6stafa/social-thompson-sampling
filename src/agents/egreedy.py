from collections.abc import Callable

import numpy as np

from .base import BaseAgent
from ..worlds.agent_data import AgentData


class EGreedyAgent(BaseAgent):
    def __init__(self, n_arms: int, epsilon: Callable[[int], float]) -> None:
        self.n_arms = n_arms
        self.epsilon = epsilon

    def reset(self) -> None:
        self.means = np.zeros(self.n_arms, dtype=float)
        self.Ts = np.zeros(self.n_arms, dtype=int)

    def get_action(self, t: int, neighbor_agents: dict[str, AgentData]) -> int:
        if np.random.uniform() <= self.epsilon(t):
            return np.random.randint(self.n_arms)
        return np.argmax(self.means)

    def update(self, t: int, action: int, reward: float, neighbor_agents: dict[str, AgentData]) -> None:
        self.means[action] = (self.means[action] * self.Ts[action] + reward) / (self.Ts[action] + 1)
        self.Ts[action] += 1
