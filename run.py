import argparse
from datetime import datetime
from multiprocessing import Pool
import pickle
from typing import Any

import numpy as np
from tqdm import tqdm

from src.utils import load_module
from src.worlds.base import BaseWorld
from src.worlds.agent_data import AgentData


# Single run function
def single_run(args):
    world_path = args

    # Load and init world
    world: BaseWorld = load_module(world_path).World()

    world.init_population()
    world.init_bandits()

    # Run
    best_rewards: list[float] = []
    last_actions: dict[str, int] = {agent_name: 0 for agent_name in world.population.nodes}
    last_rewards: dict[str, float] = {agent_name: 0 for agent_name in world.population.nodes}

    for t in range(1, world.n_trials + 1):
        bandit = world.get_bandit(t)

        _, best_reward = bandit.best_action_reward(t)
        best_rewards.append(best_reward)

        # Run all agents' action
        for agent_name in world.population.nodes:
            agent_data: AgentData = world.population.nodes[agent_name]['data']

            neighbor_agents: dict[str, AgentData] = {
                n: world.population.nodes[n]['data']
                for n in world.population.predecessors(agent_name)
            }

            action = agent_data.agent.get_action(t, neighbor_agents)
            reward = bandit.act(t, action)
            agent_data.agent.update(t, action, reward, neighbor_agents)

            last_actions[agent_name] = action
            last_rewards[agent_name] = reward

        # Update agents' log
        for agent_name in world.population.nodes:
            agent_data: AgentData = world.population.nodes[agent_name]['data']

            agent_data.actions.append(last_actions[agent_name])
            agent_data.rewards.append(last_rewards[agent_name])

            agent_data.ewm_pe.update(last_actions[agent_name])
            agent_data.ewm_pe.save_log()

    # Collect logs
    regrets: dict[str, np.ndarray] = {}
    agents_logs: dict[str, dict[str, Any]|None] = {}
    pe_logs: dict[str, list[np.ndarray]] = {}

    for agent_name in world.population.nodes:
        agent_data: AgentData = world.population.nodes[agent_name]['data']

        last_reset_index = 0
        regrets[agent_name] = np.array(best_rewards) - np.array(agent_data.rewards)
        for reset_index in world.reset_regrets + [world.n_trials]:
            regrets[agent_name][last_reset_index:reset_index] = np.cumsum(
                regrets[agent_name][last_reset_index:reset_index],
            )
            last_reset_index = reset_index

        agents_logs[agent_name] = agent_data.agent.get_logs()
        pe_logs[agent_name] = agent_data.ewm_pe.get_logs()

    return regrets, agents_logs, pe_logs


# Main
if __name__ == '__main__':
    # Get world path path from args
    parser = argparse.ArgumentParser()
    parser.add_argument('world_path')

    args = parser.parse_args()

    # Load world
    world: BaseWorld = load_module(args.world_path).World()

    # Create the log folder
    curr_datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_path = world.logs_base_path / curr_datetime_str

    logs_path.mkdir(parents=True, exist_ok=False)

    # Log objects
    all_regrets = {}
    all_agents_logs = {}
    all_pe_logs = {}

    # Run
    with tqdm(range(world.n_repeats), desc='Runs', smoothing=0) as pbar:
        with Pool(processes=world.n_processes) as pool:
            for regrets, agents_logs, pe_logs in pool.imap_unordered(
                single_run, [args.world_path] * world.n_repeats,
            ):
                for name in regrets.keys():
                    if name not in all_regrets:
                        all_regrets[name] = []
                        all_agents_logs[name] = []
                        all_pe_logs[name] = []

                    all_regrets[name].append(regrets[name])
                    if pbar.n < world.n_heavy_logs:
                        all_agents_logs[name].append(agents_logs[name])
                        all_pe_logs[name].append(pe_logs[name])

                pbar.update()

    # Calculate mean and standard error of the mean for regrets
    all_regrets_mean_sem = {}
    for name in all_regrets.keys():
        all_regrets_mean_sem[name] = (
            np.mean(all_regrets[name], axis=0),
            np.std(all_regrets[name], axis=0) / np.sqrt(world.n_repeats),
        )

    # Save logs
    with open(logs_path / 'regrets.pkl', 'wb') as f:
        pickle.dump(all_regrets_mean_sem, f)
    with open(logs_path / 'agents_logs.pkl', 'wb') as f:
        pickle.dump(all_agents_logs, f)
    with open(logs_path / 'policy_estimators_logs.pkl', 'wb') as f:
        pickle.dump(all_pe_logs, f)

    print(f'Logs are saved at {logs_path}')
