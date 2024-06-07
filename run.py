from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm

from src.policy_estimators import EWMPolicyEstimator
from src.agents import Agent, EGreedy, SocialThompsonSamplingAgent, \
    StochasticAgent, ThompsonSamplingAgent, UCB
from src.bandits import BernoulliBandit
from src.utils import exp_decay


# CONSTANTS
N = 2000  # Horizon
N_REPEATS = 1000
N_PROCESSES = 15

N_ARMS = 10
BANDIT_NS = [1000, N]
BANDIT_PS = [
    np.array([0.1, 0.175, 0.25, 0.325, 0.4, 0.475, 0.55, 0.625, 0.7, 0.9]),
    np.array([0.1, 0.175, 0.25, 0.325, 0.4, 0.475, 0.55, 0.625, 0.7, 0.9])[::-1],
]

LOGS_BASE_PATH = Path(__file__).parent.resolve() / 'logs'
NUM_LOGS = 10  # logs are saved for this number of repeats, this doesn't affect regrets

EXPERIMENTS: dict[str, list[str]] = {
    'Optimal': ['Optimal'],
    'Random': ['Random'],
    'Opponent': ['Opponent'],
    'Sub-optimal': ['Sub-optimal'],

    'TS': ['TS'],
    'TS_h25': ['TS_h25'],
    'TS_h50': ['TS_h50'],
    'TS_h100': ['TS_h100'],

    'UCB': ['UCB'],

    'EPS exp-decay': ['EPS exp-decay'],
    'EPS const': ['EPS const'],

    'Optimal + 2 Randoms + 3 Opponents': [
        'Optimal', 'Random 1', 'Random 2', 'Opponent 1', 'Opponent 2', 'Opponent 3',
    ],
    'EPS + 2 Randoms + 3 Opponents': [
        'EPS', 'Random 1', 'Random 2', 'Opponent 1', 'Opponent 2', 'Opponent 3',
    ],
    'Optimal + 2 EPSs + 3 Opponents': [
        'Optimal', 'EPS 1', 'EPS 2', 'Opponent 1', 'Opponent 2', 'Opponent 3',
    ],
}


# single run function
def single_run(args):
    i_repeat = args

    # Bandit object
    bandit = BernoulliBandit(BANDIT_NS, BANDIT_PS)
    bandit.reset()

    # Individual Agents
    optimal_agent_ps = np.full(N_ARMS, 0, dtype=float)
    optimal_agent_ps[-1] = 1.0

    sub_optimal_agent_ps = np.full(N_ARMS, 0, dtype=float)
    sub_optimal_agent_ps[-2] = 1.0

    opponent_agent_ps = np.full(N_ARMS, 0, dtype=float)
    opponent_agent_ps[0] = 1.0

    opponent_agent1_ps = np.full(N_ARMS, 0, dtype=float)
    opponent_agent1_ps[1] = 1.0

    opponent_agent2_ps = np.full(N_ARMS, 0, dtype=float)
    opponent_agent2_ps[2] = 1.0

    opponent_agent3_ps = np.full(N_ARMS, 0, dtype=float)
    opponent_agent3_ps[3] = 1.0

    optimal_agent = StochasticAgent(lambda t: optimal_agent_ps)
    sub_optimal_agent = StochasticAgent(lambda t: sub_optimal_agent_ps)

    opponent_agent = StochasticAgent(lambda t: opponent_agent_ps)
    opponent1_agent = StochasticAgent(lambda t: opponent_agent_ps)
    opponent2_agent = StochasticAgent(lambda t: opponent_agent_ps)
    opponent3_agent = StochasticAgent(lambda t: opponent_agent_ps)

    random_agent = StochasticAgent(lambda t: np.full(N_ARMS, 1/N_ARMS, dtype=float))
    random_agent1 = StochasticAgent(lambda t: np.full(N_ARMS, 1/N_ARMS, dtype=float))
    random_agent2 = StochasticAgent(lambda t: np.full(N_ARMS, 1/N_ARMS, dtype=float))

    ts_agent = ThompsonSamplingAgent(N_ARMS)
    ts_hist25_agent = ThompsonSamplingAgent(N_ARMS, history_length=25)
    ts_hist50_agent = ThompsonSamplingAgent(N_ARMS, history_length=50)
    ts_hist100_agent = ThompsonSamplingAgent(N_ARMS, history_length=100)
    ucb_agent = UCB(N, N_ARMS)

    eps_decay_func = exp_decay(init_value=1.0, decay_rate=1e-3, decay_steps=N)
    eps_decay_agent = EGreedy(eps_decay_func, N_ARMS)
    eps_agent = EGreedy(eps_decay_func, N_ARMS)
    eps1_agent = EGreedy(eps_decay_func, N_ARMS)
    eps2_agent = EGreedy(eps_decay_func, N_ARMS)
    eps_const_agent = EGreedy(lambda t: 0.3, N_ARMS)

    individual_agents: dict[str, Agent] = {
        'Optimal': optimal_agent,
        'Sub-optimal': sub_optimal_agent,

        'Random': random_agent,
        'Random 1': random_agent1,
        'Random 2': random_agent2,

        'Opponent': opponent_agent,
        'Opponent 1': opponent1_agent,
        'Opponent 2': opponent2_agent,
        'Opponent 3': opponent3_agent,

        'TS': ts_agent,
        'TS_h25': ts_hist25_agent,
        'TS_h50': ts_hist50_agent,
        'TS_h100': ts_hist100_agent,

        'UCB': ucb_agent,

        'EPS exp-decay': eps_decay_agent,
        'EPS': eps_agent,
        'EPS 1': eps1_agent,
        'EPS 2': eps2_agent,
        'EPS const': eps_const_agent,
    }

    # Social Agents & Societies
    social_agents: dict[str, Agent] = {}
    societies: dict[str, list[str]] = {}
    for e_name, society_agents in EXPERIMENTS.items():
        social_agents['STS + ' + e_name] = SocialThompsonSamplingAgent(
            N_ARMS, n_individual_agents=len(society_agents),
        )
        societies['STS + ' + e_name] = society_agents

    # regrets log
    all_agents_name = list(individual_agents.keys()) + list(social_agents.keys())
    regrets = {name: [] for name in all_agents_name}
    regrets_sum = {name: 0 for name in regrets.keys()}

    # Init policy estimators
    policy_estimators = {name: EWMPolicyEstimator(N_ARMS) for name in all_agents_name}

    # policy estimators logs
    pe_logs = {name: [] for name in policy_estimators.keys()}
    for agent_name, policy_estimator in policy_estimators.items():
        pe_logs[agent_name].append(policy_estimator.get_policy())

    # run
    for t in range(1, N + 1):
        # Individual Agents
        for agent_name in individual_agents.keys():
            action = individual_agents[agent_name].get_action(t)
            reward, best_reward = bandit.act(t, action)
            individual_agents[agent_name].update(t, action, reward)

            policy_estimators[agent_name].update(action)

            regrets_sum[agent_name] += (best_reward - reward)
            regrets[agent_name].append(regrets_sum[agent_name])

        # Social Agents
        for agent_name in social_agents.keys():
            ia_policies = np.array([policy_estimators[name].get_policy() for name in societies[agent_name]])

            action = social_agents[agent_name].get_action(t, ia_policies=ia_policies)
            reward, best_reward = bandit.act(t, action)
            social_agents[agent_name].update(t, action, reward, ia_policies=ia_policies)

            policy_estimators[agent_name].update(action)

            regrets_sum[agent_name] += (best_reward - reward)
            regrets[agent_name].append(regrets_sum[agent_name])

        for agent_name, policy_estimator in policy_estimators.items():
            pe_logs[agent_name].append(policy_estimator.get_policy())

    # Collect agents' logs
    agents_logs = {}
    for agent_name in individual_agents.keys():
        agents_logs[agent_name] = individual_agents[agent_name].get_logs()
    for agent_name in social_agents.keys():
        agents_logs[agent_name] = social_agents[agent_name].get_logs()

    return regrets, agents_logs, pe_logs


# Main
if __name__ == '__main__':
    last_n = 1
    for n, ps in zip(BANDIT_NS, BANDIT_PS):
        print(f'Bandit ps in t = [{last_n}-{n}]: {np.round(ps, 3)}')
        last_n = n + 1

    # Create the log folder
    curr_datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_path = LOGS_BASE_PATH / curr_datetime_str

    logs_path.mkdir(parents=True, exist_ok=False)

    # log objects
    all_regrets = {}
    all_agents_logs = {}
    all_pe_logs = {}

    # run
    with tqdm(range(N_REPEATS), desc='Runs') as pbar:
        with Pool(processes=N_PROCESSES) as pool:
            for regrets, agents_logs, pe_logs in pool.imap_unordered(single_run, range(N_REPEATS)):
                for name in regrets.keys():
                    if name not in all_regrets:
                        all_regrets[name] = []
                        all_agents_logs[name] = []
                        all_pe_logs[name] = []

                    all_regrets[name].append(regrets[name])
                    if pbar.n < NUM_LOGS:
                        all_agents_logs[name].append(agents_logs[name])
                        all_pe_logs[name].append(pe_logs[name])

                pbar.update()

    # calculate mean and std of regrets
    all_regrets_mean_std = {}
    for name in all_regrets.keys():
        all_regrets_mean_std[name] = (
            np.mean(all_regrets[name], axis=0),
            np.std(all_regrets[name], axis=0) / np.sqrt(N_REPEATS),
        )

    # save logs
    with open(logs_path / 'regrets.pkl', 'wb') as f:
        pickle.dump(all_regrets_mean_std, f)
    with open(logs_path / 'agents_logs.pkl', 'wb') as f:
        pickle.dump(all_agents_logs, f)
    with open(logs_path / 'policy_estimators_logs.pkl', 'wb') as f:
        pickle.dump(all_pe_logs, f)

    print(f'Logs are saved at {logs_path}')
