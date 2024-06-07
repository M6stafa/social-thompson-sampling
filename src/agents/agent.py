from typing import Any


class Agent:
    def reset(self) -> None:
        pass

    def get_action(self, t: int, *args, **kwargs) -> int:
        pass

    def update(self, t: int, action: int, reward: float, *args, **kwargs) -> None:
        pass

    def get_logs(self) -> Any:
        return None