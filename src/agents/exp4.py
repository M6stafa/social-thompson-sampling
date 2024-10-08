import random

import numpy as np

from .base import BaseAgent
from ..worlds.agent_data import AgentData


class EXP4Agent(BaseAgent):
    def __init__(self, n: int, n_arms: int, n_experts: int) -> None:
        self.n = n
        self.n_arms = n_arms
        self.n_experts = n_experts

        self.eta = np.sqrt(2 * np.log(self.n_experts) / (self.n * self.n_arms))

    def reset(self) -> None:
        self.Q = np.empty((self.n + 1, self.n_experts))
        self.Q[0] = 1 / self.n_experts

    def get_action(self, t: int, neighbor_agents: dict[str, AgentData]) -> int:
        neighbor_policies = [a.ewm_pe.get_policy() for a in neighbor_agents.values()]

        self.Pt = self.Q[t-1] @ neighbor_policies
        return random.choices(range(self.n_arms), weights=self.Pt)[0]

    def update(self, t: int, action: int, reward: float, neighbor_agents: dict[str, AgentData]) -> None:
        neighbor_policies = [a.ewm_pe.get_policy() for a in neighbor_agents.values()]

        X_hat = np.full(self.n_arms, 1)
        X_hat[action] -= (1 - reward) / (self.Pt[action] + 1e-6)

        X_tilde = neighbor_policies @ X_hat

        self.Q[t] = np.exp(self.eta * X_tilde) * self.Q[t-1]
        self.Q[t] /= np.sum(self.Q[t])
