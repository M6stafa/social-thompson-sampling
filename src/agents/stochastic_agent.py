from collections.abc import Callable
import random

import numpy as np

from .agent import Agent


class StochasticAgent(Agent):
    def __init__(self, ps: Callable[[int], list[float]|np.ndarray]) -> None:
        self.ps = ps

        self.k = len(self.ps(0))

    def get_action(self, t: int, *args, **kwargs) -> int:
        return random.choices(range(self.k), weights=self.ps(t))[0]
