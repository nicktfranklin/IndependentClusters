# cython: profile=True
# cython: linetrace=True
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

INT_DTYPE = np.int32
ctypedef np.int32_t INT_DTYPE_t

cdef extern from "math.h":
    double log(double x)

cdef extern from "math.h":
    double fmax(double a, double b)

cdef extern from "math.h":
    double abs(double x)


# @cython.wraparound(False)
# @cython.boundscheck(False)
# cpdef np.ndarray[DTYPE_t, ndim=1] value_iteration(np.ndarray[DTYPE_t, ndim=3] transition_function,
#                     np.ndarray[DTYPE_t, ndim=2] reward_function,
#                     float gamma,
#                     float iteration_criterion):
#     """
#     :param transition_function: a State x Action x State np.array of transition probabilities
#     :param reward_function: a State x State np.array of reward lotteries
#     :param gamma: discount factor (float)
#     :param iteration_criterion: small real number
#     :return: V, a np.array of State values
#     """
#     cdef int n_stimuli = transition_function.shape[0]
#     cdef int n_actions = transition_function.shape[1]
#     cdef int n_outcome = transition_function.shape[2]
#     cdef int s, a0, o
#     cdef DTYPE_t delta, v_i, v_max
#
#     cdef double[:] v = np.zeros(n_stimuli)
#     cdef double[:] v_temp = np.zeros(n_actions)
#     cdef double[:] v0 = np.zeros(n_actions)
#
#     cdef double[:, :, :] trans_f = transition_function
#     cdef double[:, :] reward_f = reward_function
#
#     # cdef doub
#     #
#     while True:
#         delta = 0
#
#         v0 = np.zeros(transition_function.shape[0], dtype=DTYPE)
#
#         for s in range(transition_function.shape[0]):
#             v_i = v[s]
#
#             # v_temp = np.zeros(n_actions, dtype=DTYPE)
#             v_temp = np.zeros(transition_function.shape[1], dtype=DTYPE)
#             v_max = -1000
#             for a0 in range(transition_function.shape[1]):
#                 for o in range(transition_function.shape[2]):
#                     v_temp[a0] += trans_f[s, a0, o] * (reward_f[s, o] + gamma * v[o])
#
#                 v_max = fmax(v_temp[a0], v_max)
#             # V0[s] = np.max(v_temp)
#             v0[s] = v_max
#             delta = fmax(delta, abs(v_i - v0[s]))
#
#         v = v0
#
#         # print delta
#         if delta < iteration_criterion:
#             return np.array(v)
cpdef np.ndarray[DTYPE_t, ndim=1] value_iteration(np.ndarray[DTYPE_t, ndim=3] transition_function,
                    np.ndarray[DTYPE_t, ndim=2] reward_function,
                    float gamma,
                    float iteration_criterion):

    cdef double [:,:,::1] trans_f = transition_function
    cdef double [:,::1] rew_f = reward_function
    cdef double [:] v = value_iteration_pure_cython(trans_f, rew_f, gamma, iteration_criterion)
    return np.asarray(v)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double [:] value_iteration_pure_cython(
        double [:,:,::1] transition_function,
        double [:, ::1] reward_function,
        float gamma,
        float iteration_criterion
):
    """
    :param transition_function: a State x Action x State np.array of transition probabilities
    :param reward_function: a State x State np.array of reward lotteries
    :param gamma: discount factor (float)
    :param iteration_criterion: small real number
    :return: V, a np.array of State values
    """
    # cdef int n_stimuli = transition_function.shape[0]
    # cdef int n_actions = transition_function.shape[1]
    cdef int n_outcome = transition_function.shape[2]
    cdef int s, a0, o
    cdef double delta, v_max, comp_value

    cdef double[:] V = np.zeros(transition_function.shape[0], dtype=DTYPE)
    cdef double[:] v_temp = np.zeros(transition_function.shape[1], dtype=DTYPE)
    # cdef double[:] v0 = np.zeros(transition_function.shape[1], dtype=DTYPE)
    cdef double[:, ::1] v_a_o = np.zeros((transition_function.shape[1], transition_function.shape[2]), dtype=DTYPE)
    # cdef double[:, :, :] trans_f = transition_function
    # cdef double[:, :] reward_f = reward_function

    # cdef doub
    #
    iteration_criterion **= 2

    cdef int t = 0
    while True:
        delta = 0
        t += 1

        # v0 = np.zeros(transition_function.shape[0], dtype=DTYPE)

        for s in range(transition_function.shape[0]):
            v = V[s]

            for a in range(transition_function.shape[1]):
                for o in range(transition_function.shape[2]):
                    v_a_o[a,o] = transition_function[s, a, o] * (reward_function[s,o] + gamma * V[o])
                v_temp = np.sum(v_a_o, axis=1)
            V[s] = np.max(v_temp)

            # np.sum()

            # v_temp = np.zeros(transition_function.shape[1], dtype=DTYPE)
            # v_max = -1000
            #
            # for a0 in range(transition_function.shape[1]):
            #     for o in range(transition_function.shape[2]):
            #         v_temp[a0] += transition_function[s, a0, o] * (reward_function[o] + gamma * V[o])
            #
            #     v_max = fmax(v_temp[a0], v_max)

            # print v, v_max

            # V[s] = v_max
            # print delta, np.abs(v-v_max), fmax(delta, abs(v - v0[s]))
            delta = fmax(delta, (v - V[s])**2)
            # print delta

        # V = v0

        # print delta
        if delta < iteration_criterion:
            # print "number itterations:  %d" % t
            # print "delta:               %f" % delta
            # print "iteration_criterion: %f" % iteration_criterion
            return V

# @cython.wraparound(False)
# @cython.boundscheck(False)
# cpdef np.ndarray[DTYPE_t, ndim=1] q_planning(np.ndarray[DTYPE_t, ndim=3] transition_function,
#                     np.ndarray[DTYPE_t, ndim=1] reward_function,
#                     float gamma,
#                     float n_reapeats):
#
#     cdef double [:,:,::1] trans_f = transition_function
#     cdef double [:] rew_f = reward_function
#     cdef double [:] v = value_iteration_pure_cython(trans_f, rew_f, gamma, iteration_criterion)
#     """
#     :param transition_function: a State x Action x State np.array of transition probabilities
#     :param reward_function: a State x State np.array of reward lotteries
#     :param gamma: discount factor (float)
#     :param iteration_criterion: small real number
#     :return: V, a np.array of State values
#     """
#
#     #select a sample at random
#     cdef int s, a
#     cdef double [:, :] q_function = np.zeros((transition_function.shape[0], transition_function.shape[1]), dtype=DTYPE)
#
#     while True:


cpdef np.ndarray[INT_DTYPE_t, ndim=1] policy_iteration(
            np.ndarray[DTYPE_t, ndim=3] transition_function,
            np.ndarray[DTYPE_t, ndim=1] reward_function,
            float gamma,
            float stop_criterion):

    cdef int n_s, n_a, s, sp, b, t, a
    n_s = transition_function.shape[0]
    n_a = transition_function.shape[1]

    cdef double [:] V = np.random.rand(n_s)
    cdef int [:] pi = np.array(np.random.randint(n_a, size=n_s), dtype=INT_DTYPE)
    cdef bint policy_stable = False
    cdef double delta, v, V_temp

    cdef double [:] rew_func = reward_function
    cdef double [:,:,::1] trans_func = transition_function
    cdef np.ndarray[DTYPE_t, ndim=1] v_a

    stop_criterion **= 2
    while not policy_stable:
        while True:
            delta = 0
            for s in range(n_s):
                v = V[s]

                # evaluate V[s] with belman eq!
                V_temp = 0
                for sp in range(n_s):
                    V_temp += trans_func[s, pi[s], sp] * (rew_func[sp] + gamma*V[sp])

                V[s] = V_temp
                delta = fmax(delta, (v - V[s])**2)

            if delta < stop_criterion:
                break

        policy_stable = True
        for s in range(n_s):
            b = pi[s]

            v_a = np.zeros(n_a, dtype=DTYPE)
            for a in range(n_a):
                for sp in range(n_s):
                    v_a[a] += trans_func[s, a, sp] * (rew_func[sp] + gamma*V[sp])

            pi[s] = np.argmax(v_a)
#
            if not b == pi[s]:
                policy_stable = False

    return np.array(pi)



cpdef np.ndarray[DTYPE_t] policy_evaluation(
        np.ndarray[INT_DTYPE_t, ndim=1] policy,
        np.ndarray[DTYPE_t, ndim=3] transition_function,
        np.ndarray[DTYPE_t, ndim=1] reward_function,
        float gamma,
        float stop_criterion):

    cdef int [:] pi = policy
    cdef double [:,:,::1] T = transition_function
    cdef double [:] R = reward_function

    cdef int n_s, sp, s
    n_s = transition_function.shape[0]
    cdef double [:] V = np.zeros(n_s, dtype=DTYPE)
    cdef double v, V_temp

    stop_criterion **= 2
    while True:
        delta = 0
        for s in range(n_s):
            v = V[s]

            V_temp = 0
            for sp in range(n_s):
                V_temp += T[s, pi[s], sp] * (R[sp] + gamma*V[sp])
            V[s] = V_temp

            delta = fmax(delta, (v - V[s])**2)

        if delta < stop_criterion:
            return np.array(V)


cpdef double get_prior_log_probability(np.ndarray[INT_DTYPE_t, ndim=1] ctx_assignment, double alpha):
    """This takes in an assignment of contexts to groups and returns the
    prior probability over the assignment using a CRP
    :param alpha:
    :param ctx_assignment:
    """
    cdef int ii, k
    cdef double log_prob = 0

    cdef int n_ctx = len(ctx_assignment)
    cdef int n_ts = len(set(ctx_assignment))
    cdef int [:] n_k = np.zeros(n_ts, dtype=INT_DTYPE)
    # cdef int[:] nk_view = n_k

    n_k[0] = 1
    for ii in range(1, n_ctx):
        k = ctx_assignment[ii]
        if n_k[k] == 0:
            log_prob += log(alpha / (np.sum(n_k) + alpha))
        else:
            log_prob += log(n_k[k] / (np.sum(n_k) + alpha))
        n_k[k] += 1

    return log_prob


# cpdef list enumerate_assignments(int max_context_number):
#     """
#      enumerate all possible assignments of contexts to clusters for a fixed number of contexts. Has the
#      hard assumption that the first context belongs to cluster #1, to remove redundant assignments that
#      differ in labeling.
#
#     :param max_context_number: int
#     :return: list of lists, each a function that takes in a context id number and returnsa cluster id number
#     """
#     cdef list _inner_loop_dict_assignments, assignments, new_list
#     cdef dict d, d_copy, dict_assignments
#     cdef np.ndarray[INT_DTYPE_t, ndim=1] cluster_assignment_function
#
#     dict_assignments = [{0: 0}]  # context 0 is always in cluster 1
#
#     for contextNumber in range(1, max_context_number):
#         _inner_loop_dict_assignments = list()
#         for d in dict_assignments:
#             new_list = list()
#             for kk in range(0, max(d.values()) + 2):
#                 d_copy = d.copy()
#                 d_copy[contextNumber] = kk
#                 new_list.append(d_copy)
#
#             _inner_loop_dict_assignments += new_list
#
#         dict_assignments = _inner_loop_dict_assignments
#
#     # turn the assignments from a dictionary of {context: cluster} to arrays where the array if a function
#     # f(context) = cluster
#     assignments = [None] * len(dict_assignments)
#     for ii, d in enumerate(dict_assignments):
#         cluster_assignment_function = np.zeros(max_context_number, dtype=INT_DTYPE)
#         for ctx_id, cluster_id in d.iteritems():
#             cluster_assignment_function[ctx_id] = cluster_id
#         assignments[ii] = cluster_assignment_function
#
#     return assignments