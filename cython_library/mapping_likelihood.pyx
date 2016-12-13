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

cpdef double negative_log_likelihood(np.ndarray[INT_DTYPE_t, ndim=1] list_actions,
                           np.ndarray[DTYPE_t, ndim=2] transition_function,
                           np.ndarray[DTYPE_t, ndim=1] mapping_function):
    """

    :param list_actions:
    :param transition_function: function of Abstract actions conditional on the observed state transition (function
     is observations by Actions)
    :param mapping_function: conditional distribution Pr(A|a)
    :return: the negative loglikelihood
    """

    cdef int A, ii
    cdef int [:] l_actions = list_actions
    cdef double [:,::1] trans_func, map_func
    trans_func = transition_function
    map_func = np.reshape(mapping_function, (4, 8))

    cdef double neg_ll = 0

    for ii in range(transition_function.shape[0]):
        for A in range(transition_function.shape[1]):
            neg_ll -= log(trans_func[ii, A] * map_func[A, l_actions[ii]])

    return neg_ll
