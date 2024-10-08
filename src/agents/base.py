from abc import abstractmethod, ABC
from typing import Any

from ..worlds.agent_data import AgentData


class BaseAgent(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_action(self, t: int, neighbor_agents: dict[str, AgentData]) -> int:
        raise NotImplementedError()

    @abstractmethod
    def update(self, t: int, action: int, reward: float, neighbor_agents: dict[str, AgentData]) -> None:
        raise NotImplementedError()

    def get_logs(self) -> dict[str, Any]|None:
        return None
