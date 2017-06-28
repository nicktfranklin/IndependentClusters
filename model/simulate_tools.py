import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import gridspec
from tqdm import tqdm

from model import make_task, JointClustering, IndependentClusterAgent, FlatControlAgent


# Define a function to Simulate the Models
def simulate_one(agent_class, simulation_number, task_kwargs, agent_kwargs=None, pruning_threshold=1000,
                 evaluate=False):

    task = make_task(**task_kwargs)
    if agent_kwargs is not None:
        agent = agent_class(task, **agent_kwargs)
    else:
        agent = agent_class(task)

    results = agent.generate(pruning_threshold=pruning_threshold, evaluate=evaluate)
    results['Simulation Number'] = [simulation_number] * len(results)
    results['Cumulative Steps Taken'] = results['n actions taken'].cumsum()

    return results


def simulate_task(n_sim, task_kwargs, agent_kwargs=None, alpha=2.0, pruning_threshold=1000, evaluate=False):
    if agent_kwargs is None:
        agent_kwargs = dict(alpha=alpha)
    elif 'alpha' not in agent_kwargs.keys():
        agent_kwargs['alpha'] = alpha

    results_jc = [None] * n_sim
    results_ic = [None] * n_sim
    results_fl = [None] * n_sim
    for ii in tqdm(range(n_sim)):
        results_jc[ii] = simulate_one(JointClustering, ii, task_kwargs, agent_kwargs=agent_kwargs,
                                      pruning_threshold=pruning_threshold, evaluate=evaluate)
        results_ic[ii] = simulate_one(IndependentClusterAgent, ii, task_kwargs, agent_kwargs=agent_kwargs,
                                      pruning_threshold=pruning_threshold, evaluate=evaluate)
        results_fl[ii] = simulate_one(FlatControlAgent, ii, task_kwargs, pruning_threshold=pruning_threshold,
                                      evaluate=evaluate)

    results_jc = pd.concat(results_jc)
    results_ic = pd.concat(results_ic)
    results_fl = pd.concat(results_fl)

    results_jc['Model'] = ['Joint Clustering'] * len(results_jc)
    results_ic['Model'] = ['Independent Clustering'] * len(results_ic)
    results_fl['Model'] = ['Flat Agent'] * len(results_fl)
    return pd.concat([results_jc, results_ic, results_fl])


def list_entropy(_list):
    h = 0
    for x in set(_list):
        p = np.sum(x == np.array(_list), dtype=float) / len(_list)
        h += -p * np.log2(p)
    return h


def mutual_information(list_a, list_b):
    h_x = list_entropy(list_a)
    h_y = list_entropy(list_b)

    h_xy = 0
    pairs = [(x, y) for x, y in zip(list_a, list_b)]
    for p in set(pairs):
        p_xy = np.sum([p == pi for pi in pairs], dtype=float) / len(pairs)
        h_xy += -p_xy * np.log2(p_xy)
    return h_x + h_y - h_xy


def plot_results(df, figsize=(6, 3)):
    with sns.axes_style('ticks'):

        _ = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.4)

        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # define the parameters to plot the results
        # cc = sns.color_palette("Set 2")
        tsplot_kwargs = dict(
            time='Trial Number',
            value='Cumulative Steps Taken',
            data=df[df['In goal']],
            unit='Simulation Number',
            condition='Model',
            estimator=np.median,
            ax=ax0,
            color="Set2",
            # order=["Independent Clustering", "Joint Clustering", "Flat Agent"]
        )

        sns.tsplot(**tsplot_kwargs)

        df0 = df[df['In goal'] & (df['Trial Number'] == df['Trial Number'].max())].copy()
        df0.loc[df0.Model == 'Independent Clustering', 'Model'] = "Independent"
        df0.loc[df0.Model == 'Joint Clustering', 'Model'] = "Joint"
        df0.loc[df0.Model == 'Flat Agent', 'Model'] = "Flat"

        sns.violinplot(data=df0, x='Model', y='Cumulative Steps Taken', ax=ax1, palette='Set2',
                       # order=["Independent", "Joint", "Flat"]
                       )
        _, ub = ax1.get_ylim()
        ax1.set_ylim([0, ub])

        sns.despine(offset=5)
