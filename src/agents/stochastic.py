from collections.abc import Callable
import random

import numpy as np

from .base import BaseAgent
from ..worlds.agent_data import AgentData


class StochasticAgent(BaseAgent):
    def __init__(self, ps: Callable[[int], list[float]|np.ndarray]) -> None:
        self.ps = ps

    def get_action(self, t: int, neighbor_agents: dict[str, AgentData]) -> int:
        ps = self.ps(t)
        return random.choices(range(len(ps)), weights=ps)[0]
