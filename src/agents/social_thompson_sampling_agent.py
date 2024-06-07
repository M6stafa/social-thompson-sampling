import numpy as np

from .agent import Agent
from .thompson_sampling_agent import ThompsonSamplingAgent
from ..policy_estimators import EWMPolicyEstimator


class SocialThompsonSamplingAgent(Agent):
    def __init__(self, k: int, n_individual_agents: int, ts_history_length: int|None = 100) -> None:
        self.k = k
        self.n_individual_agents = n_individual_agents
        self.ts_history_length = ts_history_length

        self.reset()

    def reset(self) -> None:
        # Parameters of normal dist for each agent in the form of (mean, std)
        self.agent_dists = np.array([[0.5, 1/12] for _ in range(self.n_individual_agents + 1)], dtype=float)
        self.agent_dists_log = []
        self._save_agent_dists()

        self.my_agent = ThompsonSamplingAgent(self.k, history_length=self.ts_history_length)
        self.my_policy_estimator = EWMPolicyEstimator(self.k)

    def get_action(self, t: int, ia_policies: np.ndarray, *args, **kwargs) -> int:
        selcted_agent = np.argmax([np.random.normal(loc=mean, scale=std) for mean, std in self.agent_dists])

        # My TS agent
        if selcted_agent == 0:
            return self.my_agent.get_action(t)

        # Other agents
        return np.argmax(ia_policies[selcted_agent - 1])

    def update(self, t: int, action: int, reward: float, ia_policies: np.ndarray, *args, **kwargs) -> None:
        # Update my agent policy
        self.my_agent.update(t, action, reward)
        self.my_policy_estimator.update(action)

        # Update agents' dists
        my_probs = self.my_policy_estimator.get_policy()
        select_probs = np.vstack(([my_probs], ia_policies))  # Nxk

        action_beta_dists = np.array(self.my_agent.get_beta_dists())  # kx2

        action_alphas = action_beta_dists[:, 0]
        action_betas = action_beta_dists[:, 1]
        action_sums = np.sum(action_beta_dists, axis=1)

        action_means = action_alphas / action_sums
        action_vars = action_alphas * action_betas / (np.square(action_sums) * (action_sums + 1))

        self.agent_dists[:, 0] = select_probs @ action_means
        self.agent_dists[:, 1] = np.sqrt(np.square(select_probs) @ action_vars)

        self._save_agent_dists()

    def get_logs(self):
        return {
            'agent_norm_dists': np.array(self.agent_dists_log),
            'action_beta_dists': self.my_agent.get_logs()['action_beta_dists'],
        }

    def _save_agent_dists(self) -> None:
        self.agent_dists_log.append(np.copy(self.agent_dists))
