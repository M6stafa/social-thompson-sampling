from importlib import import_module
from typing import Any


def exp_decay(init_value: float, decay_rate: float, decay_steps: int):
    def decay(step: int) -> float:
        return init_value * decay_rate ** (step / decay_steps)
    return decay


def load_module(path: str) -> Any:
    return import_module(path)
