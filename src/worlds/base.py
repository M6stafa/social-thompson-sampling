from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from os import getcwd
from pathlib import Path

import networkx as nx

from ..bandits import BaseBandit


class BaseWorld(ABC):
    n_trials: int = 2000
    n_repeats: int = 1000
    n_processes: int = cpu_count()

    logs_base_path: Path = Path(getcwd()).resolve().absolute() / 'logs'
    n_heavy_logs: int = 10
    reset_regrets = []

    n_arms: int = 10

    population: nx.DiGraph

    def __init__(self) -> None:
        self.logs_base_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def init_population(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def init_bandits(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_bandit(self, t: int) -> BaseBandit:
        raise NotImplementedError()
