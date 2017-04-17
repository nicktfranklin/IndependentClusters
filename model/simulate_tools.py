import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import make_task, JointClustering, IndependentClusterAgent, FlatControlAgent
from model import MapClusteringAgent


# Define a function to Simulate the Models
def simulate_one(agent_class, simulation_number, task_kwargs, agent_kwargs=None):
    _kwargs = copy.copy(task_kwargs)
    del _kwargs['list_goal_priors']
    task = make_task(**_kwargs)
    if agent_kwargs is not None:
        agent = agent_class(task, **agent_kwargs)
    else:
        agent = agent_class(task)

    results = agent.generate()
    results['Simulation Number'] = [simulation_number] * len(results)
    results['Cumulative Steps Taken'] = results['n actions taken'].cumsum()

    return results


def simulate_task(n_sim, task_kwargs, agent_kwargs=None, alpha=2.0):
    if agent_kwargs is None:
        agent_kwargs = dict(alpha=alpha)
    elif 'alpha' not in agent_kwargs.keys():
        agent_kwargs['alpha'] = alpha

    results_jc = [None] * n_sim
    results_ic = [None] * n_sim
    results_fl = [None] * n_sim
    for ii in tqdm(range(n_sim)):
        results_jc[ii] = simulate_one(JointClustering, ii, task_kwargs, agent_kwargs)
        results_ic[ii] = simulate_one(IndependentClusterAgent, ii, task_kwargs, agent_kwargs)
        results_fl[ii] = simulate_one(FlatControlAgent, ii, task_kwargs)

    results_jc = pd.concat(results_jc)
    results_ic = pd.concat(results_ic)
    results_fl = pd.concat(results_fl)

    results_jc['Model'] = ['Joint Clusterting'] * len(results_jc)
    results_ic['Model'] = ['Independent Clustering'] * len(results_ic)
    results_fl['Model'] = ['Flat Agent'] * len(results_fl)
    return pd.concat([results_jc, results_ic, results_fl])


def simulate_task_with_map_control(n_sim, task_kwargs, agent_kwargs=None, alpha=2.0):
    if agent_kwargs is None:
        agent_kwargs = dict(alpha=alpha)
    elif 'alpha' not in agent_kwargs.keys():
        agent_kwargs['alpha'] = alpha

    results_jc = [None] * n_sim
    results_ic = [None] * n_sim
    results_mc = [None] * n_sim
    results_fl = [None] * n_sim

    for ii in tqdm(range(n_sim)):
        results_jc[ii] = simulate_one(JointClustering, ii, task_kwargs, agent_kwargs)
        results_ic[ii] = simulate_one(IndependentClusterAgent, ii, task_kwargs, agent_kwargs)
        results_fl[ii] = simulate_one(FlatControlAgent, ii, task_kwargs)
        results_mc[ii] = simulate_one(MapClusteringAgent, ii, task_kwargs)

    results_jc = pd.concat(results_jc)
    results_ic = pd.concat(results_ic)
    results_fl = pd.concat(results_fl)
    results_mc = pd.concat(results_mc)

    results_jc['Model'] = ['Joint Clusterting'] * len(results_jc)
    results_ic['Model'] = ['Independent Clustering'] * len(results_ic)
    results_fl['Model'] = ['Flat Agent'] * len(results_fl)
    results_mc['Model'] = ['Map Clustering'] * len(results_mc)

    return pd.concat([results_jc, results_ic, results_fl, results_mc])


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