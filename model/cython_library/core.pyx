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

            if not b == pi[s]:
                policy_stable = False

    return np.array(pi)

cpdef np.ndarray[DTYPE_t] value_iteration(
        np.ndarray[DTYPE_t, ndim=3] transition_function,
        np.ndarray[DTYPE_t, ndim=1] reward_function,
        float gamma,
        float stop_criterion):

    cdef double [:,:,::1] T = transition_function
    cdef double [:] R = reward_function

    cdef int n_s, sp, s, a, n_a
    n_s = transition_function.shape[0]
    n_a = transition_function.shape[1]
    cdef double [:] V = np.zeros(n_s, dtype=DTYPE)
    cdef double v, V_temp, v_sa_temp

    stop_criterion **= 2
    while True:
        delta = 0
        for s in range(n_s):
            v = V[s]

            V_temp = 0
            # get the value of each action, then take the max value of the action for the state
            for a in range(n_a):
                v_sa_temp = 0
                for sp in range(n_s):
                    v_sa_temp += T[s, a, sp] * (R[sp] + gamma * V[sp])
                V_temp = fmax(v_sa_temp, V_temp)

            V[s] = V_temp

            delta = fmax(delta, (v - V[s])**2)

        if delta < stop_criterion:
            return np.array(V)


cpdef double get_prior_log_probability(dict ctx_assignment, double alpha):
    """This takes in an assignment of contexts to groups and returns the
    prior probability over the assignment using a CRP
    :param alpha:
    :param ctx_assignment:
    """
    cdef int ii, k
    cdef double log_prob = 0

    cdef int n_ctx = len(ctx_assignment.keys())
    cdef int K
    if len(ctx_assignment) > 0:
        K = max(ctx_assignment.values()) + 1
    else:
        K = 1
    cdef int [:] n_k = np.zeros(K, dtype=INT_DTYPE)

    # n_k[0] = 1
    for ii in ctx_assignment.keys():
        k = ctx_assignment[ii]
        if n_k[k] == 0:
            log_prob += log(alpha / (np.sum(n_k) + alpha))
        else:
            log_prob += log(n_k[k] / (np.sum(n_k) + alpha))
        n_k[k] += 1

    return log_prob
