from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents import BaseAgent
from ..policy_estimators import EWMPolicyEstimator


@dataclass
class AgentData:
    agent: BaseAgent
    actions: list[int] = field(default_factory=lambda: [])
    rewards: list[float] = field(default_factory=lambda: [])
    ewm_pe: EWMPolicyEstimator = None
