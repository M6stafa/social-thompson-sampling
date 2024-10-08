import numpy as np

from .base import BaseAgent
from ..worlds.agent_data import AgentData


class UCBAgent(BaseAgent):
    def __init__(self, n_arms: int, exploration_factor: float = 1) -> None:
        self.n_arms = n_arms
        self.exploration_factor = exploration_factor

        self.reset()

    def reset(self) -> None:
        self.means = np.zeros(self.n_arms, dtype=float)
        self.Ts = np.zeros(self.n_arms, dtype=int)
        self.ucbs = np.full(self.n_arms, np.inf)

    def get_action(self, t: int, neighbor_agents: dict[str, AgentData]) -> int:
        ucbs = np.where(
            self.Ts == 0,
            np.inf,
            self.means + self.exploration_factor * np.sqrt(np.log(t) / (self.Ts + 1e-6)),
        )

        return np.argmax(ucbs)

    def update(self, t: int, action: int, reward: float, neighbor_agents: dict[str, AgentData]) -> None:
        self.means[action] = (self.means[action] * self.Ts[action] + reward) / (self.Ts[action] + 1)
        self.Ts[action] += 1
