import numpy as np
import copy


def sample_cmf(cmf):
    return int(np.sum(np.random.rand() > cmf))


def softmax_to_pdf(q_values, inverse_temperature):
    pdf = np.exp(np.array(q_values) * float(inverse_temperature))
    pdf = pdf / np.sum(pdf)
    return pdf


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


def randomize_order(context_balance, hazard_rates):
    context_order = []
    context_presentations = copy.copy(context_balance)

    n_ctx = len(context_presentations)
    n_trials = 0
    for n_rep in context_presentations:
        n_trials += n_rep

    # randomly select the first context
    available_contexts = []
    for ctx in range(n_ctx):
        for jj in range(context_presentations[ctx]):
            available_contexts.append(ctx)

    current_context = available_contexts[np.random.randint(len(available_contexts))];
    context_presentations[current_context] -= 1

    n_repeats = 0

    for ii in range(n_trials):
        context_order.append(current_context)

        # determine if there is to be a context switch
        if (np.random.rand() < hazard_rates[n_repeats]) | (context_presentations[current_context] < 1):

            # construct a list of available contexts to select
            _available_ctx = range(n_ctx)
            available_contexts = []
            for ctx in _available_ctx:
                if (context_presentations[ctx] > 0) & (ctx != current_context):
                    available_contexts += [ctx] * context_presentations[ctx]

            # randomly select one available context
            if available_contexts: # check if empty!
                current_context = available_contexts[np.random.randint(len(available_contexts))]
                n_repeats = -1

        # update counters
        n_repeats += 1
        context_presentations[current_context] -= 1

    return context_order