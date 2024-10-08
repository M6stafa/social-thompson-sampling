from abc import ABC, abstractmethod


class BaseBandit(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def act(self, t: int, action: int) -> float:
        # Return reward of the selected action
        raise NotImplementedError()

    @abstractmethod
    def best_action_reward(self, t: int) -> tuple[int, float]:
        # Return the best action and its reward in trial `t`
        raise NotImplementedError()
