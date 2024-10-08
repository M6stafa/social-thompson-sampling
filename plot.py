import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from tqdm import tqdm


# CONSTANTS
PLOTS_BASE_PATH = Path(__file__).parent.resolve() / 'plots'


# helper functions
def hex_to_rgba(h, alpha):
    return f'rgba{tuple([int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)] + [alpha])}'


def create_lines_fig(
    x,
    means_stds: dict[str, tuple[np.ndarray, np.ndarray]],
    title='',
    xaxis_title='',
    yaxis_title='',
    x_max: int|None = None,
    reduce_x: int = 0,
):
    x_max = len(x) if x_max is None else x_max
    x = x[:x_max]

    if reduce_x > 0:
        x_indexes = np.arange(0, len(x), reduce_x, dtype=int).tolist()
    else:
        x_indexes = np.arange(len(x), dtype=int).tolist()
    if x_indexes[-1] != (len(x) - 1):
        x_indexes.append(len(x) - 1)

    x = np.take(x, x_indexes).tolist()

    color_i = 0
    colors = px.colors.qualitative.Plotly

    fig = go.Figure()

    for name, (means, stds) in means_stds.items():
        cis = 1.96 * stds

        line_color = hex_to_rgba(colors[color_i], 1.0)
        fill_color = hex_to_rgba(colors[color_i], 0.2)

        fig.add_traces([
            go.Scatter(
                x = x,
                y = means,
                line = dict(color=line_color),
                mode = 'lines',
                name = name,
                legendgroup = name,
            ),
            go.Scatter(
                x = x + x[::-1],
                y = list(means + cis) + list(means - cis)[::-1],
                fill = 'toself',
                fillcolor = fill_color,
                line = dict(color='rgba(0, 0, 0, 0)', dash='dot'),
                hoverinfo = 'skip',
                showlegend = False,
                legendgroup = name,
            ),
            go.Scatter(
                x = x,
                y = list(means + cis),
                line = dict(color=fill_color, dash='dot'),
                hoverinfo = 'skip',
                showlegend = False,
                legendgroup = name,
            ),
            go.Scatter(
                x = x,
                y = list(means - cis),
                line = dict(color=fill_color, dash='dot'),
                hoverinfo = 'skip',
                showlegend = False,
                legendgroup = name,
            ),
        ])

        color_i += 1
        color_i %= len(colors)

    fig.update_layout(
        title = title,
        xaxis_title = xaxis_title,
        yaxis_title = yaxis_title,
    )

    return fig


# Get logs base path from cmd
parser = argparse.ArgumentParser()
parser.add_argument('logs_dir')

args = parser.parse_args()

logs_path = Path(args.logs_dir).resolve(strict=True).absolute()

# plots path
plots_path = PLOTS_BASE_PATH / logs_path.name
plots_path.mkdir(parents=True, exist_ok=True)


# Regrets
print('Ploting regrets...')
with open(logs_path / 'regrets.pkl', 'rb') as f:
    regrets = pickle.load(f)

# TODO: read horizon from config
fig = create_lines_fig(np.arange(1000) + 1, regrets)

fig.update_layout(
    xaxis_title = 't',
    yaxis_title = 'Expected regret',
)
fig.write_html(plots_path / 'regrets.html', include_plotlyjs='directory')


# Policy Estimators
print('Ploting policy estimators...')
with open(logs_path / 'policy_estimators_logs.pkl', 'rb') as f:
    pe_logs = pickle.load(f)

log_steps = 4
pe_data = []
for agent_name, policy_estimations in pe_logs.items():
    for t, hist in enumerate(policy_estimations[0]):
        if t % log_steps != 0:
            continue
        for action, prob in enumerate(hist):
            pe_data.append((agent_name, t, action, prob))

df_pe = pd.DataFrame(pe_data, columns=['Agent', 't', 'Action', 'Probability'])

fig = px.bar(
    df_pe,
    x = 'Action',
    y = 'Probability',
    animation_frame = 't',
    color = 'Agent',
    barmode = 'group',
    range_y = [0, 1],
    title = 'Policy Estimators',
)
fig.update_xaxes(type='category')
fig.write_html(plots_path / 'policy_estimators.html', auto_play=False, include_plotlyjs='directory')


# plot agent logs
with open(logs_path / 'agents_logs.pkl', 'rb') as f:
    agents_logs = pickle.load(f)

log_steps = 5
n_samples = 10_000
x_dist = np.linspace(0, 1, num=1000)

# Agents' dist
for agent_name, agent_logs in tqdm(agents_logs.items(), desc='Ploting agents\' dist'):
    agent_logs = agent_logs[0]

    if (type(agent_logs) is not dict) or ('agent_norm_dists' not in agent_logs):
        continue

    agent_norm_dists: list[dict[str, tuple[float, float]]] = agent_logs['agent_norm_dists']

    data_pdf = []
    data_selection_probs = []

    for t in np.linspace(0, len(agent_norm_dists) - 1, num=len(agent_norm_dists) // log_steps, dtype=int):
        agent_dists_names = list(agent_norm_dists[t].keys())
        new_samples = []

        for i, name in enumerate(agent_dists_names):
            mean, std = agent_norm_dists[t][name]
            agent_dist = stats.norm(loc=mean, scale=std)

            pdf_values = agent_dist.pdf(x_dist)

            for x, v in zip(x_dist, pdf_values):
                data_pdf.append((name, t + 1, x, v))

            new_samples.append(agent_dist.rvs(size=n_samples))

        selection_probs = np.bincount(np.argmax(new_samples, axis=0), minlength=len(agent_dists_names))
        selection_probs = selection_probs / np.sum(selection_probs)

        for name, prob in zip(agent_dists_names, selection_probs):
            data_selection_probs.append((name, t + 1, prob))

    df_pdf = pd.DataFrame(data_pdf, columns=['Agent', 't', 'x', 'PDF'])
    fig = px.line(
        df_pdf,
        x = 'x',
        y = 'PDF',
        animation_frame = 't',
        color = 'Agent',
        range_x = [0, 1],
        range_y = [-1, np.max(df_pdf['PDF']) + 1],
        title = agent_name + ' (Select Agent Norm Dists)',
    )
    fig.write_html(
        plots_path / f'agent_norm_dists {agent_name}.html',
        auto_play = False,
        include_plotlyjs = 'directory',
    )

    df_selection_probs = pd.DataFrame(
        data_selection_probs,
        columns = ['Agent', 't', 'Probability of selecting agent'],
    )
    fig = px.line(
        df_selection_probs,
        x = 't',
        y = 'Probability of selecting agent',
        color = 'Agent',
        range_y = [-0.01, 1.01],
        title = agent_name + ' (Agent Selection Probabilities)',
    )
    fig.write_html(
        plots_path / f'agent_probs {agent_name}.html',
        auto_play = False,
        include_plotlyjs = 'directory',
    )


# Action's dist
for agent_name, agent_logs in tqdm(agents_logs.items(), desc='Ploting actions\' dist'):
    agent_logs = agent_logs[0]

    if (type(agent_logs) is not dict) or ('action_beta_dists' not in agent_logs):
        continue

    action_beta_dists = agent_logs['action_beta_dists']
    action_dists_names = [str(i) for i in range(len(action_beta_dists[0]))]

    data_pdf = []
    data_selection_probs = []

    for t in np.linspace(0, len(action_beta_dists) - 1, num=len(action_beta_dists) // log_steps, dtype=int):
        new_samples = []

        for i, name in enumerate(action_dists_names):
            a, b = action_beta_dists[t][i]
            action_dist = stats.beta(a, b)

            pdf_values = action_dist.pdf(x_dist)

            for x, v in zip(x_dist, pdf_values):
                data_pdf.append((name, t + 1, x, v))

            new_samples.append(action_dist.rvs(size=n_samples))

        selection_probs = np.bincount(np.argmax(new_samples, axis=0), minlength=len(action_dists_names))
        selection_probs = selection_probs / np.sum(selection_probs)

        for name, prob in zip(action_dists_names, selection_probs):
            data_selection_probs.append((name, t + 1, prob))

    df_pdf = pd.DataFrame(data_pdf, columns=['Action', 't', 'x', 'PDF'])
    fig = px.line(
        df_pdf,
        x = 'x',
        y = 'PDF',
        animation_frame = 't',
        color = 'Action',
        range_x = [0, 1],
        range_y = [-1, np.max(df_pdf['PDF']) + 1],
        title = agent_name + ' (Select Action Beta Dists)',
    )
    fig.write_html(
        plots_path / f'action_dists {agent_name}.html',
        auto_play = False,
        include_plotlyjs = 'directory',
    )

    df_selection_probs = pd.DataFrame(
        data_selection_probs,
        columns = ['Action', 't', 'Probability of selecting action'],
    )
    fig = px.line(
        df_selection_probs,
        x = 't',
        y = 'Probability of selecting action',
        color = 'Action',
        range_y = [-0.01, 1.01],
        title = agent_name + ' (Action Selection Probabilities)',
    )
    fig.write_html(
        plots_path / f'action_probs {agent_name}.html',
        auto_play = False,
        include_plotlyjs = 'directory',
    )
