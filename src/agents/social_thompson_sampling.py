from typing import Any

import numpy as np

from .base import BaseAgent
from .thompson_sampling import ThompsonSamplingAgent
from ..policy_estimators import EWMPolicyEstimator
from ..worlds.agent_data import AgentData


class SocialThompsonSamplingAgent(BaseAgent):
    def __init__(
        self,
        n_arms: int,
        ts_history_length: int|None = 100,
        ewm_lambda: float = 2 / (20 + 1),
    ) -> None:
        self.n_arms = n_arms
        self.ts_history_length = ts_history_length
        self.ewm_lambda = ewm_lambda

    def reset(self) -> None:
        self._agent_norm_dists: list[dict[str, tuple[float, float]]] = []

        self.my_agent = ThompsonSamplingAgent(self.n_arms, history_length=self.ts_history_length)
        self.my_policy_estimator = EWMPolicyEstimator(self.n_arms, self.ewm_lambda)

    def get_action(self, t: int, neighbor_agents: dict[str, AgentData]) -> int:
        # Update agents' dists
        my_policy = self.my_agent.get_policy()
        other_policies = [a.ewm_pe.get_policy() for a in neighbor_agents.values()]

        select_probs = np.vstack(([my_policy], other_policies))  # Nxk
        action_beta_dists = np.array(self.my_agent.get_beta_dists())  # kx2

        action_alphas = action_beta_dists[:, 0]
        action_betas = action_beta_dists[:, 1]
        action_sums = np.sum(action_beta_dists, axis=1)

        action_means = action_alphas / action_sums
        action_vars = action_alphas * action_betas / (np.square(action_sums) * (action_sums + 1))

        agent_means = select_probs @ action_means
        agent_stds = np.sqrt(np.square(select_probs) @ action_vars)

        # Save norm dists
        self._agent_norm_dists.append({})
        for agent_name, mean, std in zip(['STS'] + list(neighbor_agents.keys()), agent_means, agent_stds):
            self._agent_norm_dists[-1][agent_name] = (mean, std)

        # Select agent with thompson sampling
        self._selcted_agent = np.argmax([
            np.random.normal(loc=mean, scale=std)
            for mean, std in zip(agent_means, agent_stds)
        ])

        # Return action of selected agent
        if self._selcted_agent == 0:
            return self.my_agent.get_action(t, None)
        return np.argmax(other_policies[self._selcted_agent - 1])

    def update(self, t: int, action: int, reward: float, neighbor_agents: dict[str, AgentData]) -> None:
        # Update my agent
        self.my_agent.update(t, action, reward, None)
        self.my_policy_estimator.update(action)

    def get_logs(self) -> dict[str, Any]|None:
        return {
            'agent_norm_dists': np.array(self._agent_norm_dists),
            'action_beta_dists': self.my_agent.get_logs()['action_beta_dists'],
        }
