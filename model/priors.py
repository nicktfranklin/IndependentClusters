import numpy as np
from numpy.random import randint

"""
These objects are prior that can be used to generate random tasks

"""


def sample_cmf(cmf):
    return int(np.sum(np.random.rand() > cmf))


class Prior(object):

    def __init__(self, n_goals, n_mappings, alpha):
        self.n_goals = n_goals
        self.n_maps = n_mappings
        self.alpha = float(alpha)

    def update(self, goal_idx, map_idx):
        pass

    def sample(self):
        return 0, 0


class JointPrior(Prior):

    def __init__(self, n_goals=4, n_mappings=2, alpha=1.0):
        super(JointPrior, self).__init__(n_goals, n_mappings, alpha)
        self.cluster_counts = dict()
        self.cluster_probability = dict()
        self.A = self.alpha

    def update(self, goal_idx, map_idx):
        # update the counts
        if (goal_idx, map_idx) in self.cluster_counts.keys():
            self.cluster_counts[(goal_idx, map_idx)] += 1
        else:
            self.cluster_counts[(goal_idx, map_idx)] = 1

        # update the normalizing constant
        self.A = self.alpha
        for N_k in self.cluster_counts.values():
            self.A += N_k

        # update the cluster probabilities
        for k, N_k in self.cluster_counts.iteritems():
            self.cluster_probability[k] = N_k / self.A

    def sample(self):

        # create the cmf as a vector
        k_idx = dict()
        N_K = len(self.cluster_probability)
        pmf = np.zeros(N_K + 1)
        for ii, (k, p) in enumerate(self.cluster_probability.iteritems()):
            k_idx[ii] = k
            pmf[ii] = p

        pmf[N_K] = self.alpha / self.A
        cmf = np.cumsum(pmf)

        # sample the cmf
        k = sample_cmf(cmf)

        # if old cluster return the goal and mapping id,
        # otherwise generate a random new combination
        if k < N_K:
            return k_idx[k]
        else:
            return randint(self.n_goals), randint(self.n_maps)


class IndependentPrior(Prior):

    def __init__(self, n_goals=4, n_mappings=2, alpha=1.0):
        super(IndependentPrior, self).__init__(n_goals, n_mappings, alpha)
        self.goal_cluster_counts = dict()
        self.goal_cluster_probability = dict()
        self.mapping_cluster_counts = dict()
        self.mapping_cluster_probability = dict()
        self.A_g = self.alpha
        self.A_m = self.alpha

    def update(self, goal_idx, map_idx):
        # update the goal counts
        if goal_idx in self.goal_cluster_counts.keys():
            self.goal_cluster_counts[goal_idx] += 1
        else:
            self.goal_cluster_counts[goal_idx] = 1

        # update the mapping counts
        if map_idx in self.mapping_cluster_counts.keys():
            self.mapping_cluster_counts[map_idx] += 1
        else:
            self.mapping_cluster_counts[map_idx] = 1

        # update the normalizing constants
        self.A_g = self.alpha
        for N_k_g in self.goal_cluster_counts.values():
            self.A_g += N_k_g

        self.A_m = self.alpha
        for N_k_m in self.mapping_cluster_counts.values():
            self.A_m += N_k_m

        # update the cluster probabilities
        for k_g, N_k_g in self.goal_cluster_counts.iteritems():
            self.goal_cluster_probability[k_g] = N_k_g / self.A_g

        for k_m, N_k_m in self.mapping_cluster_probability.iteritems():
            self.mapping_cluster_probability[k_m] = N_k_m / self.A_m

    def sample(self):

        # create each cmf as a vector
        N_K_g = len(self.goal_cluster_probability)
        pmf_g = np.zeros(N_K_g + 1)
        k_idx_g = dict()
        for ii, (k_g, p) in enumerate(self.goal_cluster_probability.iteritems()):
            k_idx_g[ii] = k_g
            pmf_g[ii] = p

        N_K_m = len(self.mapping_cluster_probability)
        pmf_m = np.zeros(N_K_m + 1)
        k_idx_m = dict()
        for ii, (k_m, p) in enumerate(self.mapping_cluster_probability.iteritems()):
            k_idx_m[ii] = k_m
            pmf_m[ii] = p

        pmf_g[N_K_g] = self.alpha / self.A_g
        pmf_m[N_K_m] = self.alpha / self.A_m
        cmf_g = np.cumsum(pmf_g)
        cmf_m = np.cumsum(pmf_m)

        # sample the cmfs
        k_g = sample_cmf(cmf_g)
        k_m = sample_cmf(cmf_m)

        # if old cluster return the goal and mapping id,
        # otherwise generate a random new combination
        if k_g < N_K_g:
            k_g = k_idx_g[k_g]
        else:
            k_g = randint(self.n_goals)
        if k_m < N_K_m:
            k_m = k_idx_m[k_m]
        else:
            k_m = randint(self.n_maps)

        return k_g, k_m

