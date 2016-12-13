import numpy as np


def sample_cmf(cmf):
    return int(np.sum(np.random.rand() > cmf))


def softmax_to_pdf(q_values, inverse_temperature):
    pdf = np.exp(np.array(q_values) * float(inverse_temperature))
    pdf = pdf / np.sum(pdf)
    return pdf

def value_iteration(transition_function, reward_function, gamma, stop_criterion):
    n_s = transition_function.shape[0]
    n_a = transition_function.shape[1]
    V = np.random.rand(n_s)

    while True:
        delta = 0
        for s in range(n_s):
            v = V[s]

            v_temp = np.zeros(n_a)
            for a in range(n_a):
                for sp in range(n_s):
                    v_temp[a] += transition_function[s, a, sp] * (reward_function[s, sp] + gamma*V[s])
            V[s] = np.max(v_temp)

            delta = np.max([delta, np.abs(v-V[s])])
        if delta < stop_criterion:
            return V


def policy_iteration(transition_function, reward_function, gamma, stop_criterion):

    n_s = transition_function.shape[0]
    n_a = transition_function.shape[1]
    V = np.random.rand(n_s)
    pi = np.random.randint(n_a, size=9)

    policy_stable = False
    while policy_stable == False:
        while True:
            delta = 0
            for s in range(n_s):
                v = V[s]

                # evaluate V[s] with belman eq!
                V_temp = 0
                for sp in range(n_s):
                    V_temp += transition_function[s, pi[s], sp] * (reward_function[s, sp] + gamma*V[sp])

                V[s] = V_temp
                delta = np.max([delta, np.abs(v - V[s])])

            if delta < stop_criterion:
                break

        policy_stable = True
        for s in range(n_s):
            b = pi[s]

            v_a = np.zeros(n_a)
            for a in range(n_a):
                for sp in range(n_s):
                    v_a[a] += transition_function[s, a, sp] * (reward_function[s, sp] + gamma*V[sp])

            pi[s] = np.argmax(v_a)
#
            if not b == pi[s]:
                policy_stable = False
    return pi


def policy_evaluation(policy, transition_function, reward_function, gamma, stop_criterion):
    n_s = transition_function.shape[0]
    V = np.zeros(n_s)

    while True:
        delta = 0
        for s in range(n_s):
            v = V[s]

            V_temp = 0
            for sp in range(n_s):
                V_temp += transition_function[s, policy[s], sp] * (reward_function[s, sp] + gamma*V[sp])
            V[s] = V_temp

            delta = np.max([delta, np.abs(v - V[s])])

        if delta < stop_criterion:
            return V


def enumerate_assignments(max_context_number):
    """
     enumerate all possible assignments of contexts to clusters for a fixed number of contexts. Has the
     hard assumption that the first context belongs to cluster #1, to remove redundant assignments that
     differ in labeling.

    :param max_context_number: int
    :return: list of lists, each a function that takes in a context id number and returnsa cluster id number
    """
    dict_assignments = [{0: 0}]  # context 0 is always in cluster 1

    for contextNumber in range(1, max_context_number):
        __inner_loop_dict_assignments = list()
        for d in dict_assignments:
            new_list = list()
            for kk in range(0, max(d.values()) + 2):
                d_copy = d.copy()
                d_copy[contextNumber] = kk
                new_list.append(d_copy)

            __inner_loop_dict_assignments += new_list

        dict_assignments = __inner_loop_dict_assignments

    # turn the assignments from a dictionary of {context: cluster} to arrays where the array if a function
    # f(context) = cluster
    assignments = [None] * len(dict_assignments)
    for ii, d in enumerate(dict_assignments):
        cluster_assignment_function = np.zeros(max_context_number, dtype=np.int32)
        for ctx_id, cluster_id in d.iteritems():
            cluster_assignment_function[ctx_id] = cluster_id
        assignments[ii] = cluster_assignment_function

    return assignments
