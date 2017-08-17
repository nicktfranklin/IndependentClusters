import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import list_entropy, mutual_information

sns.set_style('ticks')
sns.set_palette('Set2')


class CRP(object):

    def __init__(self, alpha, max_goals=4):
        self.alpha = alpha
        self.cluster_assignments = dict()
        self.max_goals = max_goals

    def update(self, c, k):
        self.cluster_assignments[c] = k

    def predict(self):
        """ generate a cluster prediction for context n+1"""
        t = len(self.cluster_assignments)
        pmf = np.zeros(t+1)
        for k in set(self.cluster_assignments.values()):
            N_k = np.sum(np.array(self.cluster_assignments.values()) == k)
            pmf[k] = N_k / (t + self.alpha)
        pmf[-1] = self.alpha / (t + self.alpha)
        return pmf / pmf.sum()

    def predict_goal(self):
        pmf = self.predict()

        # number of goals seen
        n_g = min(len(self.cluster_assignments.values()), self.max_goals)
        chance_prob = 1 - pmf[:n_g].sum()

        goal_prediction_pmf = np.zeros(self.max_goals)
        for g in range(n_g):
            goal_prediction_pmf[g] = pmf[g]
        goal_prediction_pmf += (chance_prob / float(self.max_goals))

        return goal_prediction_pmf


def relabel(context_goals):
    visited_goals = dict()
    relabeled = list()
    g_max = 0
    for g in context_goals:
        if g not in visited_goals.keys():
            visited_goals[g] = g_max
            g_max += 1
        g0 = visited_goals[g]
        relabeled.append(g0)
    return relabeled


def make_dict(context_goals):
    _context_goals = relabel(context_goals)
    return {c: k for c, k in enumerate(_context_goals)}


def kl_divergence(context_goals, pmf, alpha=1.0):
    context_goal_pairs = make_dict(context_goals)
    crp = CRP(alpha=alpha)

    goal_guesses = 0
    n_guesses = 0
    for c, k in context_goal_pairs.iteritems():
        crp_pmf = crp.predict_goal()
        goal_guesses += np.log2(pmf[k] / crp_pmf[k])
        crp.update(c, k)
        n_guesses += 1
    return goal_guesses / n_guesses


def evaluate(context_goals, alpha=1.0, max_goals=4):
    context_goal_pairs = make_dict(context_goals)
    crp = CRP(alpha=alpha, max_goals=max_goals)

    goal_guesses = 0
    n_guesses = 0
    for c, k in context_goal_pairs.iteritems():
        crp_pmf = crp.predict_goal()
        goal_guesses -= np.log2(crp_pmf[k])
        crp.update(c, k)
        n_guesses += 1
    return goal_guesses / n_guesses


def evaluate_joint(context_goals, context_maps, alpha=1.0, max_goals=4):
    context_goal_pairs = make_dict(context_maps)
    context_map_pairs = make_dict(context_goals)

    crps = [CRP(alpha=alpha, max_goals=max_goals) for _ in set(context_map_pairs.values())]
    goal_guesses = 0
    n_guesses = 0
    for m, crp in enumerate(crps):
        # create a list of goals conditional on each map
        context_goal_pairs = make_dict(
            [g for g, m0 in zip(context_goals, context_maps) if m0 == m]
        )

        for c, k in context_goal_pairs.iteritems():
            crp_pmf = crp.predict_goal()
            goal_guesses -= np.log2(crp_pmf[k])
            crp.update(c, k)
            n_guesses += 1


    return goal_guesses / n_guesses


def evaluate_onezero(context_goals, alpha=1.0, max_goals=4):
    context_goal_pairs = make_dict(context_goals)
    crp = CRP(alpha=alpha, max_goals=max_goals)

    goal_guesses = 0
    n_guesses = 0
    for c, k in context_goal_pairs.iteritems():
        crp_pmf = crp.predict_goal()
        goal_guesses += 1-crp_pmf[k]
        crp.update(c, k)
        n_guesses += 1
    return goal_guesses / n_guesses


def upperbound_onezero(context_goals):
    p = (np.bincount(context_goals) / float(np.shape(context_goals)[0]))
    error = 0.0
    for c in context_goals:
        error += 1-p[c]

    return error / len(context_goals)


def upperbound2_onezero(context_goals):
    p = (np.bincount(context_goals) / float(np.shape(context_goals)[0]))
    error = 0.0
    for c in context_goals:
        error += float(p[c] == np.max(p))

    return error / len(context_goals)

def plot_evaluate(context_goals, max_goals=4):
    print "Entropy H(Goal):           %.2f" % list_entropy(context_goals)
    plt.figure(figsize=(8, 5))
    x = np.arange(0.10, 10.0, 0.05)
    h = [evaluate(context_goals, alpha=x0, max_goals=max_goals) for x0 in x]
    handle_ind, = plt.plot(x, h, label='Goal Clustering')

    chance_probability = np.log2(max_goals)
    handle_ub, = plt.plot([0, 10], [chance_probability, chance_probability], 'k:',
            label='Uniform Guess over Goals')

    lower_bound = list_entropy(context_goals)
    handle_lb0, = plt.plot([0, 10], [lower_bound, lower_bound], 'k--',
            label='Entropy of Pr(Goal)')
    ax = plt.gca()
    ax.set_position([0.1,0.1,0.5,0.8])
    plt.legend(handles=[handle_ind, handle_ub, handle_lb0], loc='center left',
               bbox_to_anchor = (1.0, 0.5))
    _, ub = ax.get_ylim()
    ax.set_ylim([0, ub*1.1])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Entropy (bits)')
    sns.despine()
    return ax


def plot_evaluate_onezero(context_goals, max_goals=4):
    print "Entropy H(Goal):           %.2f" % list_entropy(context_goals)
    plt.figure(figsize=(8, 5))
    x = np.arange(0.10, 10.0, 0.05)
    h = [evaluate_onezero(context_goals, alpha=x0, max_goals=max_goals) for x0 in x]
    handle_ind, = plt.plot(x, h, label='Goal Clustering')

    chance_probability = 1.0/max_goals
    handle_ub, = plt.plot([0, 10], [chance_probability, chance_probability], 'k:',
            label='Uniform Guess over Goals')

    upper_bound = upperbound_onezero(context_goals)
    handle_lb0, = plt.plot([0, 10], [upper_bound, upper_bound], 'k--',
            label='Generative Distribution')

    upper_bound = upperbound2_onezero(context_goals)
    handle_lb1, = plt.plot([0, 10], [upper_bound, upper_bound], 'k--',
            label='Guess Max Distribution')

    ax = plt.gca()
    ax.set_position([0.1,0.1,0.5,0.8])
    plt.legend(handles=[handle_ind, handle_ub, handle_lb0, handle_lb1], loc='center left',
               bbox_to_anchor = (1.0, 0.5))
    _, ub = ax.get_ylim()
    ax.set_ylim([0, ub*1.1])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Entropy (bits)')
    sns.despine()
    return ax

def plot_evaluate_joint(context_goals, context_maps, max_goals=4):
    print "Entropy H(Goal):           %.2f" % list_entropy(context_goals)
    print "Entropy H(Enviornment):    %.2f" % list_entropy(context_maps)
    print "Mutual Information I(G,E): %.2f" % mutual_information(context_goals, context_maps)

    plt.figure(figsize=(8, 5))
    x = np.arange(0.10, 10.0, 0.05)
    h_ind = [evaluate(context_goals, alpha=x0, max_goals=max_goals) for x0 in x]
    h_joint = [evaluate_joint(context_goals, context_maps, alpha=x0, max_goals=max_goals)
              for x0 in x]
    handle_ind, = plt.plot(x, h_ind, label='Independent Goal Clustering')
    handle_joint, = plt.plot(x, h_joint, 'r', label='Conditional Goal Clustering')

    chance_probability = np.log2(max_goals)
    handle_ub, = plt.plot([0, 10], [chance_probability, chance_probability], 'k:',
            label='Uniform Guess over Goals)')

    independent_lower_bound = list_entropy(context_goals)
    handle_lb0, = plt.plot([0, 10], [independent_lower_bound, independent_lower_bound], 'k--',
            label='Entropy of Pr(Goal)')

    conditional_lower_bound = list_entropy(context_goals) - \
            mutual_information(context_goals, context_maps)

    handle_lb1, = plt.plot([0, 10], [conditional_lower_bound, conditional_lower_bound], 'k',
            label='Entropy of Pr(Goal|Envoirnment)')
    ax = plt.gca()
    ax.set_position([0.1,0.1,0.5,0.8])
    plt.legend(handles=[handle_ind, handle_joint, handle_ub, handle_lb0, handle_lb1],
               loc='center left', bbox_to_anchor = (1.0, 0.5))
    _, ub = ax.get_ylim()
    ax.set_ylim([0, ub*1.1])
    sns.despine()
    return ax
