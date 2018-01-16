#boundscheck=False, wraparound=True
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

from core import value_iteration
from core import get_prior_log_probability

DTYPE = np.float
ctypedef np.float_t DTYPE_t

INT_DTYPE = np.int32
ctypedef np.int32_t INT_DTYPE_t

cdef extern from "math.h":
    double log(double x)



cdef class TransitionCluster(object):
    cdef double [:,:,::1] transition_counts, pmf
    cdef int n_actions, n_states
    cdef double prior

    def __init__(self, int n_actions, int n_states, float prior):


        # instantiate the transition function as an SxAxS matrix expressing the functions
        # p(s'|a, s). This can be calculated analytically by keeping an empirical count of 
        # transitions.

        cdef double[:, :, ::1] transition_counts

        transition_counts = np.zeros((n_states, n_actions, n_states), dtype=float)

        # initialize the counts with the prior
        cdef int s, a
        for s in range(n_states):
            for a in range(n_actions):
                transition_counts[s, a, s] = prior

        self.transition_counts = transition_counts

        # calculate the conditional probabilty function -- this is needed for the likelihood and 
        # precaluating is faster. Here, is just initialization to P(s'=s|a, s) = 1 i.e. each state is 
        # self absorbing with probabilty 1.0
        cdef double[:, :, ::1] pmf = np.zeros((n_states, n_actions, n_states), dtype=float)
        cdef int sp
        cdef float k
        for s in range(n_states):
            for a in range(n_actions):
                pmf[s, a, s] = 1.0
        self.pmf = pmf

        self.n_actions = n_actions
        self.n_states = n_states
        self.prior = prior

    def update(self, int s, int a, int sp):
        # just update the counts
        self.transition_counts[s, a, sp] += 1.0
        self._update_pmf()

    def _update_pmf(self):
        # precalculate the normalizing constant
        cdef int s, a, sp
        cdef double k
        for s in range(self.n_states):
            for a in range(self.n_actions):
                k = 0.0
                for sp in range(self.n_states):
                    k += self.transition_counts[s, a, sp]

                for sp in range(self.n_states):
                    self.pmf[s, a, sp] = self.transition_counts[s, a, sp] / k

    def get_pmf(self, int s, int a):
        # p(s'|a, s)
        cdef int sp
        cdef double[::1] pmf = np.zeros(self.n_states, dtype=float)
        for sp in range(self.n_states):
            pmf[sp] = self.pmf[s, a, sp]

        return pmf

    def get_likelihood(self, int s, int a, int sp):
        # p(s'|a, s)
        return self.pmf[s, a, sp]

    def deep_copy(self):
        cdef TransitionCluster _cluster_copy = TransitionCluster(self.n_actions, self.n_states, self.prior)

        cdef int s, a, sp
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for sp in range(self.n_states):
                    _cluster_copy.transition_counts[s, a, sp] = self.transition_counts[s, a, sp]
        _cluster_copy._update_pmf()

        return _cluster_copy

    def get_transition_function(self):
        return np.asarray(self.pmf)


cdef class TransitionHypothesis(object):

        cdef dict cluster_assignments, clusters
        cdef double prior_log_prob, alpha, prior
        cdef list experience
        cdef int n_actions, n_states

        def __init__(self, int n_actions, int n_states, float alpha, float prior):

            self.n_actions = n_actions
            self.n_states = n_states
            self.alpha = alpha
            self.prior = prior

            # initialize clusters
            self.clusters = dict()
            self.cluster_assignments = dict()

            # store the prior probability
            self.prior_log_prob = 0

            # need to store all experiences for log probability calculations
            self.experience = list()

        cdef _update_prior(self):
            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)

        def deep_copy(self):
            cdef TransitionHypothesis _h_copy = TransitionHypothesis(self.n_actions, self.n_states,
                                                               self.alpha, self.prior)

            cdef int k, s, a, sp, c
            cdef TransitionCluster cluster

            # deep copy each list, dictionary, cluster, etc
            _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
            _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
            _h_copy.experience = [(k, s, a, sp) for k, s, a, sp in self.experience]
            _h_copy.prior_log_prob = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)
            return _h_copy

        def add_new_context_assignment(self, int c, int k):
            """
            :param c: context id number
            :param k: cluster id number
            :return:
            """
            # check if new cluster
            if k not in self.cluster_assignments.values():
                self.clusters[k] = TransitionCluster(self.n_actions, self.n_states, self.prior)

            self.cluster_assignments[c] = k  # note, there's no check built in here
            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)

        def get_assignments(self):
            return self.cluster_assignments

        def update(self, int c, int s, int a, sp):
            cdef int k = self.cluster_assignments[c]
            cdef TransitionCluster cluster = self.clusters[k]
            cluster.update(s, a, sp)
            self.clusters[k] = cluster

            # need to store all experiences for log probability calculations
            self.experience.append((k, s, a, sp))

        def get_log_likelihood(self):
            cdef double log_likelihood = 0
            cdef int k, s, a, sp
            cdef TransitionCluster cluster

            #loop through experiences and get posterior
            for k, s, a, sp in self.experience:
                cluster = self.clusters[k]
                log_likelihood += log(cluster.get_likelihood(s, a, sp))

            return log_likelihood

        # def get_visit_total(self):


        def get_log_posterior(self):
            return self.prior_log_prob + self.get_log_likelihood()

        def get_transition_probability(self, int c, int s, int a):
            cdef TransitionCluster cluster = self.clusters[self.cluster_assignments[c]]
            return cluster.get_pmf(s, a)

        def get_transition_function(self, c):
            cdef TransitionCluster cluster = self.clusters[self.cluster_assignments[c]]
            return np.asarray(cluster.get_transition_function())

        def get_log_prior(self):
            return self.prior_log_prob