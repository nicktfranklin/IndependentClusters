# cython: profile=True, linetrace=True, boundscheck=False, wraparound=True
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

from core import policy_iteration, policy_evaluation
from core import get_prior_log_probability

DTYPE = np.float
ctypedef np.float_t DTYPE_t

INT_DTYPE = np.int32
ctypedef np.int32_t INT_DTYPE_t

cdef extern from "math.h":
    double log(double x)

cdef class Hypothesis(object):

    cdef double get_log_posterior(self):
        pass


cdef class MappingCluster(object):
    cdef double [:,::1] mapping_history, mapping_mle, pr_aa_given_a
    cdef double [:] abstract_action_counts, primitive_action_counts
    cdef int n_primitive_actions, n_abstract_actions

    def __init__(self, int n_primitive_actions, int n_abstract_actions, float mapping_prior):

        cdef double[:, ::1] mapping_history, mapping_mle, pr_aa_given_a
        cdef double[:] abstract_action_counts, primitive_action_counts

        mapping_history = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=float) * mapping_prior
        abstract_action_counts = np.ones(n_abstract_actions+1, dtype=float) *  mapping_prior * n_primitive_actions
        mapping_mle = np.ones((n_primitive_actions, n_abstract_actions + 1),  dtype=float) * \
                      (1.0 / n_primitive_actions)

        primitive_action_counts = np.ones(n_primitive_actions, dtype=DTYPE) * mapping_prior * n_abstract_actions
        pr_aa_given_a = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE) * \
                        (1.0 / n_abstract_actions)

        self.mapping_history = mapping_history
        self.abstract_action_counts = abstract_action_counts
        self.mapping_mle = mapping_mle
        self.primitive_action_counts = primitive_action_counts
        self.pr_aa_given_a = pr_aa_given_a

        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

    def update(self, int a, int aa):
        cdef int aa0, a0
        self.mapping_history[a, aa] += 1.0
        self.abstract_action_counts[aa] += 1.0
        self.primitive_action_counts[a] += 1.0

        for aa0 in range(self.n_abstract_actions):
            for a0 in range(self.n_primitive_actions):
                self.mapping_mle[a0, aa0] = self.mapping_history[a0, aa0] / self.abstract_action_counts[aa0]

                # p(A|a, k) estimator
                self.pr_aa_given_a[a0, aa0] = self.mapping_history[a0, aa0] / self.primitive_action_counts[a0]

    def get_mapping_mle(self, int a, int aa):
        return self.mapping_mle[a, aa]

    def get_likelihood(self, int a, int aa):
        return self.pr_aa_given_a[a, aa]

cdef class MappingHypothesis(object):

        cdef dict cluster_assignments, clusters
        cdef double prior_log_prob, alpha, mapping_prior
        cdef list experience
        cdef int n_abstract_actions, n_primitive_actions

        def __init__(self, int n_primitive_actions, int n_abstract_actions,
                     dict cluster_assignments, float alpha, float mapping_prior):

            cdef int n_k = len(set(cluster_assignments))
            self.cluster_assignments = cluster_assignments
            self.n_primitive_actions = n_primitive_actions
            self.n_abstract_actions = n_abstract_actions
            self.alpha = alpha
            self.mapping_prior = mapping_prior

            # initialize mapping clusters
            self.clusters = {k: MappingCluster(n_primitive_actions, n_abstract_actions, mapping_prior)
                             for k in cluster_assignments.keys()}

            # store the prior probability
            self.prior_log_prob = get_prior_log_probability(cluster_assignments, alpha)

            # need to store all experiences for log probability calculations
            self.experience = list()

        cdef _update_prior(self):
            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)

        cdef _get_cluster_average(self):
            pass

        cdef update_assignment(self, int c, int k):
            # if we are assigning a context to a previous cluster, it's pretty simple,
            # otherwise we need a new set of statistics
            if k in self.cluster_assignments.values():
                self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions, self.mapping_prior)
            self.cluster_assignments[c] = k


        def updating_mapping(self, int c, int a, int aa):
            cdef int k = self.cluster_assignments[c]
            cdef MappingCluster cluster = self.clusters[k]
            cluster.update(a, aa)
            self.clusters[k] = cluster

            # need to store all experiences for log probability calculations
            self.experience.append((k, a, aa))

        def get_log_likelihood(self):
            cdef double log_likelihood = 0
            cdef int t, k, a, aa
            cdef MappingCluster cluster

            #loop through experiences and get posterior
            for k, a, aa in self.experience:
                cluster = self.clusters[k]
                log_likelihood += log(cluster.get_likelihood(a, aa))

            return log_likelihood

        def get_log_posterior(self):
            return self.prior_log_prob + self.get_log_likelihood()

        def get_mapping_probability(self, int c, int a, int aa):
            cdef MappingCluster cluster = self.clusters[self.cluster_assignments[c]]
            return cluster.get_mapping_mle(a, aa)

        def get_log_prior(self):
            return self.prior_log_prob


cdef class RewardHypothesis(object):
    cdef double gamma, iteration_criterion, log_prior, inverse_temperature
    cdef int n_stim
    cdef dict cluster_assignments, clusters
    cdef double [:,::1] reward_visits, reward_received, reward_function, reward_received_bool
    cdef double [:,:,::1] reward_probability
    cdef list experience

    def __init__(self, int n_stim, float inverse_temp, float gamma, float stop_criterion,
                 dict cluster_assignments, float alpha):

        self.n_stim = n_stim
        self.inverse_temperature = inverse_temp
        self.gamma = gamma
        self.iteration_criterion = stop_criterion
        self.cluster_assignments = cluster_assignments

        cdef int n_k = len(set(cluster_assignments))

        # rewards!
        self.reward_visits = np.ones((n_k, n_stim)) * 0.011
        self.reward_received = np.ones((n_k, n_stim)) * 0.01
        self.reward_function = np.ones((n_k, n_stim)) * (0.01/0.011)

        # need a separate tracker for the probability a reward was received
        self.reward_received_bool = np.ones((n_k, n_stim)) * 1e-5
        self.reward_probability = np.ones((n_k, n_stim, 2)) * (1e-5/2e-5)

        # initialize posterior
        self.experience = list()
        self.log_prior = get_prior_log_probability(cluster_assignments, alpha)

    def update(self, int c, int sp, int r):
        cdef int k = self.cluster_assignments[c]
        self.reward_visits[k, sp] += 1.0
        self.reward_received[k, sp] += r
        self.reward_function[k, sp] = self.reward_received[k, sp] / self.reward_visits[k, sp]

        self.reward_received_bool[k, sp] += float(r > 0)
        self.reward_probability[k, sp, 1] = self.reward_received_bool[k, sp] / self.reward_visits[k, sp]
        self.reward_probability[k, sp, 0] = 1 - self.reward_probability[k, sp, 1]

        self.experience.append((k, sp, r))

    cpdef double get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int t, k, sp, r
        for k, sp, r in self.experience:
            t = int(r>0)
            log_likelihood += log(self.reward_probability[k, sp, t])
        return log_likelihood

    def get_log_posterior(self):
        return self.get_log_likelihood() + self.log_prior

    def get_log_prior(self):
        return self.log_prior

    cpdef np.ndarray[DTYPE_t, ndim=1] get_abstract_action_q_values(self, int s, int c, double[:,:,::1] transition_function):
        cdef int k = self.cluster_assignments[c]
        pi = policy_iteration(np.asarray(transition_function), np.asarray(self.reward_function[k, :]), gamma=self.gamma,
                              stop_criterion=self.iteration_criterion)

        v = policy_evaluation(
            pi,
            np.asarray(transition_function),
            np.asarray(self.reward_function[k, :]),
            gamma=self.gamma,
            stop_criterion=self.iteration_criterion
        )

        n_abstract_actions = np.shape(transition_function)[1]

        # use the bellman equation to solve the q_values
        q_values = np.zeros(n_abstract_actions)
        cdef int aa0, sp0
        for aa0 in range(n_abstract_actions):
            for sp0 in range(self.n_stim):
                q_values[aa0] += transition_function[s, aa0, sp0] * (self.reward_function[k, sp0] + self.gamma * v[sp0])

        return np.array(q_values)

    def select_abstract_action_pmf(self, int s, int c, double[:,:,::1] transition_function):
        cdef np.ndarray[DTYPE_t, ndim=1] q_values = self.get_abstract_action_q_values(s, c, transition_function)

        # we need q-values to properly consider multiple options of equivalent optimality, but we can just always
        # pass a very high value for the temperature
        cdef np.ndarray[DTYPE_t, ndim=1] pmf = np.exp(np.array(q_values) * float(self.inverse_temperature))
        pmf = pmf / np.sum(pmf)

        return pmf


    def get_reward_function(self, int c):
        cdef int ts = self.cluster_assignments[c]
        cdef np.ndarray[DTYPE_t, ndim=1] reward_function = np.zeros(self.n_stim, dtype=np.float)

        for s in range(self.n_stim):
            reward_function[s] = self.reward_function[ts, s]
        return reward_function
    
    def get_reward_visits(self, int c):
        cdef int k = self.cluster_assignments[c]
        cdef np.ndarray[DTYPE_t, ndim=1] reward_visits = np.zeros(self.n_stim, dtype=np.float)

        for s in range(self.n_stim):
            reward_visits[s] = self.reward_visits[k, s]
        return reward_visits

    def set_reward_prior(self, list list_goals):
        cdef int n_k, n_stim, ts, s
        n_k, n_stim = np.shape(self.reward_visits)

        # rewards!
        self.reward_visits = np.ones((n_k, n_stim)) * 0.0001
        self.reward_received = np.ones((n_k, n_stim)) * 0.00001

        for ts in range(n_k):
            for s in list_goals:
                self.reward_visits[ts, s] += 0.001
                self.reward_received[ts, s] += 0.001

        for ts in range(n_k):
            for s in range(n_stim):
                self.reward_function[ts, s] = self.reward_received[ts, s] / self.reward_visits[ts, s]
