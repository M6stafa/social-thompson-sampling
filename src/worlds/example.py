import networkx as nx
import numpy as np

from .base import BaseWorld
from .agent_data import AgentData
from ..agents import SocialThompsonSamplingAgent, StochasticAgent, \
    ThompsonSamplingAgent, UCBAgent
from ..bandits import BaseBandit, BernoulliBandit
from ..policy_estimators import EWMPolicyEstimator


class World(BaseWorld):
    n_trials: int = 1000
    n_repeats: int = 100

    n_arms: int = 10

    ewm_lambda: float = 2 / (20 + 1)
    ts_history_lengths = [None, 100]
    sts_history_lengths = [100]

    def init_population(self):
        G = nx.DiGraph(name='Example World')

        # Stocahstic agents
        optimal_agent_ps = np.full(self.n_arms, 0, dtype=float)
        optimal_agent_ps[-1] = 1.0
        G.add_node('Optimal', data=AgentData(
            agent = StochasticAgent(lambda t: optimal_agent_ps),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))

        sub_optimal_agent_ps = np.full(self.n_arms, 0, dtype=float)
        sub_optimal_agent_ps[-2] = 1.0
        G.add_node('Sub-optimal', data=AgentData(
            agent = StochasticAgent(lambda t: sub_optimal_agent_ps),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))

        opponent_agent_ps = np.full(self.n_arms, 0, dtype=float)
        opponent_agent_ps[0] = 1.0
        G.add_node('Opponent', data=AgentData(
            agent = StochasticAgent(lambda t: opponent_agent_ps),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))
        G.add_node('Opponent 1', data=AgentData(
            agent = StochasticAgent(lambda t: opponent_agent_ps),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))
        G.add_node('Opponent 2', data=AgentData(
            agent = StochasticAgent(lambda t: opponent_agent_ps),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))
        G.add_node('Opponent 3', data=AgentData(
            agent = StochasticAgent(lambda t: opponent_agent_ps),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))

        G.add_node('Random', data=AgentData(
            agent = StochasticAgent(lambda t: np.full(self.n_arms, 1/self.n_arms, dtype=float)),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))
        G.add_node('Random 1', data=AgentData(
            agent = StochasticAgent(lambda t: np.full(self.n_arms, 1/self.n_arms, dtype=float)),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))
        G.add_node('Random 2', data=AgentData(
            agent = StochasticAgent(lambda t: np.full(self.n_arms, 1/self.n_arms, dtype=float)),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))

        # Individual learner agents
        for h in self.ts_history_lengths:
            G.add_node(f'TS (h={h})', data=AgentData(
                agent = ThompsonSamplingAgent(self.n_arms, history_length=h),
                ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
            ))

        for i in range(3):
            G.add_node(f'TS {i+1} (h={self.ts_history_lengths[-1]})', data=AgentData(
                agent = ThompsonSamplingAgent(self.n_arms, history_length=self.ts_history_lengths[-1]),
                ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
            ))

        G.add_node('UCB', data=AgentData(
            agent = UCBAgent(self.n_arms, 1),
            ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
        ))

        # Social agents
        social_experiments: dict[str, list[str]] = {
            'Optimal': ['Optimal'],
            'Random': ['Random'],
            'Opponent': ['Opponent'],
            'Sub-optimal': ['Sub-optimal'],

            **{
                f'TS (h={h})': [f'TS (h={h})']
                for h in self.ts_history_lengths
            },

            **{
                f'{i} TSs': [f'TS {j+1} (h={self.ts_history_lengths[-1]})' for j in range(i)]
                for i in range(2, 4)
            },

            'UCB': ['UCB'],

            'Optimal + 2 Randoms + 3 Opponents': [
                'Optimal', 'Random 1', 'Random 2', 'Opponent 1', 'Opponent 2', 'Opponent 3',
            ],
        }

        for e_name, society_agents in social_experiments.items():
            # Social Thompson Sampling
            for h in self.sts_history_lengths:
                sts_agent_name = f'STS (h={h}) + {e_name}'
                G.add_node(sts_agent_name, data=AgentData(
                    agent = SocialThompsonSamplingAgent(self.n_arms, ts_history_length=h),
                    ewm_pe = EWMPolicyEstimator(self.n_arms, self.ewm_lambda),
                ))

                for agent_name in society_agents:
                    G.add_edge(agent_name, sts_agent_name)

        self.population = G

    def init_bandits(self) -> None:
        init_ps = np.array([0.1, 0.175, 0.25, 0.325, 0.4, 0.475, 0.55, 0.625, 0.7, 0.9])
        self.bandit_ns = [self.n_trials // 2, self.n_trials]
        self.reset_regrets = [self.n_trials // 2]
        bandit_ps = [init_ps, init_ps[::-1]]
        # bandit_ps = [init_ps, np.roll(init_ps, 1)]

        self.bandits = []
        for ps in bandit_ps:
            self.bandits.append(BernoulliBandit(self.n_trials, ps))

    def get_bandit(self, t: int) -> BaseBandit:
        bandit_arg = 0
        while t > self.bandit_ns[bandit_arg]:
            bandit_arg += 1

        return self.bandits[bandit_arg]
