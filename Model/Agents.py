import numpy as np
import pandas as pd

from GridWorld import Task, get_goal_guess_sequence
from cython_library import RewardHypothesis, MappingHypothesis
from cython_library import policy_iteration, policy_evaluation

""" these agents differ from the generative agents I typically use in that I need to pass a transition
function (and possibly a reward function) to the agent for each trial. """


def make_q_primitive(q_abstract, mapping):
    q_primitive = np.zeros(8)
    n, m = np.shape(mapping)
    for aa in range(m):
        for a in range(n):
            q_primitive[a] += q_abstract[aa] * mapping[a, aa]
    return q_primitive


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


def softmax_to_pdf(q_values, inverse_temperature):
    pdf = np.exp(np.array(q_values) * float(inverse_temperature))
    pdf = pdf / np.sum(pdf)
    return pdf


def sample_cmf(cmf):
    return int(np.sum(np.random.rand() > cmf))


def displacement_to_abstract_action(dx, dy):
    if (not dx == 0) & (not dy == 0):
        return -1

    if dx == 1:
        return 0
    elif dx == -1:
        return 1
    elif dy == 1:
        return 2
    elif dy == -1:
        return 3
    else:
        return -1


class MultiStepAgent(object):

    def __init__(self, task):
        self.task = task
        assert type(self.task) is Task
        self.current_trial = 0

    def get_action_pmf(self, state):
        return np.ones(self.task.n_primitive_actions, dtype=float) / self.task.n_primitive_actions

    def get_action_cmf(self, state):
        return np.cumsum(self.get_action_pmf(state))

    def select_action(self, state):
        return sample_cmf(self.get_action_cmf(state))

    def update(self, experience_tuple):
        pass

    def get_optimal_policy(self, state):
        pass

    def get_value_function(self, state):
        pass

    def get_reward_function(self, state):
        pass

    def generate(self):
        """ run through all of the trials of a task and output trial-by-trial data
        :return:
        """

        # count the number of steps to completion
        step_counter = np.zeros(self.task.n_trials)
        results = list()
        times_seen_ctx = np.zeros(self.task.n_ctx)

        ii = 0
        while True:

            # get the current state and evaluate stop condition
            state = self.task.get_state()
            if state is None:
                break

            t = self.task.current_trial_number
            step_counter[t] += 1

            _, c = state

            # select an action
            action = self.select_action(state)

            # save for data output
            action_map = self.task.current_trial.action_map
            goal_location = self.task.current_trial.goal_location
            walls = self.task.current_trial.walls
            inverse_abstract_action_key = self.task.current_trial.inverse_abstract_action_key

            # take an action
            experience_tuple = self.task.move(action)
            ((x, y), c), a, aa, r, ((xp, yp), _) = experience_tuple

            # update the learner
            self.update(experience_tuple)

            if step_counter[t] == 1:
                times_seen_ctx[c] += 1

            # if step_counter[t] == 100:
            #     self.task.current_trial_number += 1
            #     if self.task.current_trial_number < len(self.task.trials):
            #         self.task.current_trial = self.task.trials[self.task.current_trial_number]
            #     else:
            #         self.task.current_trial = None

            results.append(pd.DataFrame({
                'Start Location': [(x, y)],
                'End Location': [(xp, yp)],
                'context': [c],
                'key-press': [action],
                'action': [inverse_abstract_action_key[aa]],  # the cardinal movement, in words
                'Reward Collected': [r],
                'n actions taken': step_counter[t],
                'TrialNumber': [t],
                'In goal': not (self.task.current_trial_number == t),
                'Times Seen Context': times_seen_ctx[c],
                'action_map': [action_map],
                'goal location': [goal_location],
                'walls': [walls]
            }, index=[ii]))

            ii += 1

        return pd.concat(results)


class FullInformationAgent(MultiStepAgent):
    """ this agent uses the reward function and transition function to solve the task exactly.
    """

    def __init__(self, task, discount_rate=0.8, iteration_criterion=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions

    def _transitions_to_actions(self, s, sp):
        x, y = self.task.inverse_state_loc_key[s]
        xp, yp = self.task.inverse_state_loc_key[sp]
        return displacement_to_abstract_action(xp - x, yp - y)

    def get_value_function(self, state):
        ''' this particluar funcito ingornse thes state b/c it is full information
        :param state:
        :return:
        '''

        # run policy iteration on known reward and transition function
        pi = policy_iteration(self.task.current_trial.transition_function,
                              self.task.current_trial.reward_function[:, :],
                              gamma=self.gamma,
                              stop_criterion=self.iteration_criterion)

        # use policy to get value function
        v = policy_evaluation(pi,
                              self.task.current_trial.transition_function,
                              self.task.current_trial.reward_function[:, :],
                              gamma=self.gamma,
                              stop_criterion=self.iteration_criterion)

        return v

    def get_abstract_q_values(self, state):

        ((x, y), c) = state
        s = self.task.state_location_key[(x, y)]

        value_function = self.get_value_function(state)
        trial = self.task.current_trial

        q_abstract = np.zeros(self.n_abstract_actions, dtype=float)
        for aa0 in range(self.n_abstract_actions):
            for o0 in range(self.task.n_states):
                q_abstract[aa0] += trial.transition_function[s, aa0, o0] * (self.reward_function[c, s, o0] +
                                                                            self.gamma * value_function[o0])

        return q_abstract

    def select_abstract_action(self, state):
        (x, y), c = state

        # what is current state?
        s = self.task.state_location_key[(x, y)]

        pi = policy_iteration(self.task.current_trial.transition_function,
                              self.task.current_trial.reward_function[s, :],
                              gamma=self.gamma,
                              stop_criterion=self.iteration_criterion)

        # use the policy to choose the correct action for the current state
        abstract_action = pi[s]

        return abstract_action

    def select_action(self, state):
        (x, y), c = state

        abstract_action = self.select_abstract_action(state)
        # print "Abstract Action Selected: ", abstract_action, _abstract_action_to_displacement(abstract_action)

        # use the actual action_mapping to get the correct primitive action key
        inverse_abstract_action_key = {aa: move for move, aa in self.task.current_trial.abstract_action_key.iteritems()}
        inverse_action_map = {move: key_press for key_press, move in self.task.current_trial.action_map.iteritems()}
        key_press_to_primitive = {key_press: ii for ii, key_press in enumerate(self.task.primitive_actions)}

        move = inverse_abstract_action_key[abstract_action]
        # print move, inverse_abstract_action_key, inverse_action_map, key_press_to_primitive
        # print inverse_action_map
        key_press = inverse_action_map[move]
        return key_press
        # a = key_press_to_primitive[key_press]

        # return a


class FlatAgentKnownRewards(FullInformationAgent):
    """ This Agent uses the known reward function (from the task) and estimates the mapping
    """

    def __init__(self, task, discount_rate=0.8, iteration_criterion=0.01, epsilon=0.025):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.epsilon = epsilon

        # mappings!
        self.mapping_history = np.ones((self.task.n_ctx, self.task.n_primitive_actions, self.task.n_abstract_actions+1),
                                        dtype=float) * 0.01
        self.mapping_mle = np.ones((self.task.n_ctx, self.task.n_primitive_actions, self.task.n_abstract_actions),
                                   dtype=float) * (1.0/self.task.n_primitive_actions)

        self.abstract_action_counts = np.ones((self.task.n_ctx, self.task.n_abstract_actions+1), dtype=float) * 0.01 * \
            self.task.n_primitive_actions

    def update(self, experience_tuple):
        self.updating_mapping(experience_tuple)

    def updating_mapping(self, experience_tuple):
        # print experience_tuple
        (_, c), a, aa, r, _ = experience_tuple

        self.mapping_history[c, a, aa] += 1.0
        self.abstract_action_counts[c, aa] += 1.0

        for aa0 in range(self.task.n_abstract_actions):
            for a0 in range(self.task.n_primitive_actions):
                self.mapping_mle[c, a0, aa0] = self.mapping_history[c, a0, aa0] / self.abstract_action_counts[c, aa0]

    def _transitions_to_actions(self, s, sp):
        x, y = self.task.inverse_state_loc_key[s]
        xp, yp = self.task.inverse_state_loc_key[sp]
        return displacement_to_abstract_action(xp - x, yp - y)

    def select_abstract_action(self, state):
        (x, y), c = state
        # use epsilon greedy choice function
        if np.random.rand() > self.epsilon:
            pi = policy_iteration(self.task.current_trial.transition_function,
                                  self.task.current_trial.reward_function[:, :],
                                  gamma=self.gamma,
                                  stop_criterion=self.iteration_criterion)

            #
            s = self.task.state_location_key[(x, y)]
            abstract_action = pi[s]
        else:
            abstract_action = np.random.randint(self.n_abstract_actions)

        return abstract_action

    def select_action(self, state):
        _, c = state

        # use epsilon greedy choice function:
        if np.random.rand() > self.epsilon:
            abstract_action = self.select_abstract_action(state)

            # use the mapping estimate to create a pmf over the primitives
            pmf = self.mapping_mle[c, :, abstract_action]

            # the mapping estimator is P(A|a), not P(a|A). This is a bit of a fudge, so normalize to create pmf
            pmf /= pmf.sum()
            cmf = pmf.cumsum()

            pmf = self.mapping_mle[c, :, abstract_action]
            for aa0 in range(self.task.n_abstract_actions):
                if not aa0 == abstract_action:
                    pmf *= (1 - self.mapping_mle[c, :, aa0])

            pmf /= pmf.sum()

            return sample_cmf(cmf)
        else:
            return np.random.randint(self.task.primitive_actions)


class FlatAgent(FullInformationAgent):
    """ This Agent learns the reward function and mapping will model based planning
    """

    def __init__(self, task, discount_rate=0.8, iteration_criterion=0.01, mapping_prior=0.01, epsilon=0.025):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions
        self.epsilon = epsilon

        # mappings!
        self.mapping_history = np.ones((self.task.n_ctx, self.task.n_primitive_actions, self.task.n_abstract_actions+1),
                                        dtype=float) * mapping_prior
        self.abstract_action_counts = np.ones((self.task.n_ctx, self.task.n_abstract_actions+1), dtype=float) * \
                                      mapping_prior * self.task.n_primitive_actions
        self.mapping_mle = np.ones((self.task.n_ctx, self.task.n_primitive_actions, self.task.n_abstract_actions),
                                   dtype=float) * (1.0/self.task.n_primitive_actions)

        # rewards!
        self.reward_visits = np.ones((self.task.n_ctx, self.task.n_states)) * 0.0001
        self.reward_received = np.ones((self.task.n_ctx, self.task.n_states)) * 0.001
        self.reward_function = np.ones((self.task.n_ctx, self.task.n_states)) * (0.001/0.0101)

    def update(self, experience_tuple):
        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

    def update_rewards(self, c, sp, r):
        self.reward_visits[c, sp] += 1.0
        self.reward_received[c, sp] += r
        self.reward_function[c, sp] = self.reward_received[c, sp] / self.reward_visits[c, sp]

    def updating_mapping(self, c, a, aa):

        self.mapping_history[c, a, aa] += 1.0
        self.abstract_action_counts[c, aa] += 1.0

        for aa0 in range(self.task.n_abstract_actions):
            for a0 in range(self.task.n_primitive_actions):
                self.mapping_mle[c, a0, aa0] = self.mapping_history[c, a0, aa0] / self.abstract_action_counts[c, aa0]

    def _transitions_to_actions(self, s, sp):
        x, y = self.task.inverse_state_loc_key[s]
        xp, yp = self.task.inverse_state_loc_key[sp]
        return displacement_to_abstract_action(xp - x, yp - y)

    def select_abstract_action(self, state):

        # use epsilon greedy choice function
        if np.random.rand() > self.epsilon:
            (x, y), c = state
            pi = policy_iteration(self.task.current_trial.transition_function,
                                  self.reward_function[c, :],
                                  gamma=self.gamma,
                                  stop_criterion=self.iteration_criterion)

            #
            s = self.task.state_location_key[(x, y)]
            abstract_action = pi[s]
        else:
            abstract_action = np.random.randint(self.n_abstract_actions)

        return abstract_action

    def select_action(self, state):

        # use epsilon greedy choice function
        if np.random.rand() > self.epsilon:
            _, c = state

            abstract_action = self.select_abstract_action(state)

            pmf = self.mapping_mle[c, :, abstract_action]
            for aa0 in range(self.task.n_abstract_actions):
                if not aa0 == abstract_action:
                    pmf *= (1 - self.mapping_mle[c, :, aa0])

            pmf /= pmf.sum()

            return sample_cmf(pmf.cumsum())
        else:
            return np.random.randint(self.n_primitive_actions)

    def set_reward_prior(self, list_locations):
        """
        This method allows the agent to specific grid coordinates as potential goal locations by
        putting some prior (low confidence) reward density over the grid locations.

        All other locations have low reward probability

        :param list_locations: a list of (x, y) coordinates to consider as priors for the goal location search
        :return: None
        """
        # this makes for a 10% reward received prior over putative non-goal states
        self.reward_visits = np.ones((self.task.n_ctx, self.task.n_states)) * 0.0001
        self.reward_received = np.ones((self.task.n_ctx, self.task.n_states)) * 0.00001

        for loc in list_locations:
            s = self.task.state_location_key[loc]
            self.reward_received[:, s] += 0.001
            self.reward_visits[:, s] += 0.001

        for s in range(self.task.n_states):
            for c in range(self.task.n_ctx):
                self.reward_function[c, s] = self.reward_received[c, s] / self.reward_visits[c, s]


class FlatAgentQValues(FlatAgent):
    def get_abstract_q_values(self, state):
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        pi = policy_iteration(self.task.current_trial.transition_function,
                              self.reward_function[c, :],
                              gamma=self.gamma,
                              stop_criterion=self.iteration_criterion)

        v = policy_evaluation(pi,
                              self.task.current_trial.transition_function,
                              self.reward_function[c, :],
                              gamma=self.gamma,
                              stop_criterion=self.iteration_criterion)

        # Use the belman equation to get q-values
        q_abstract = np.zeros(self.task.n_abstract_actions, dtype=float)
        for aa0 in range(self.task.n_abstract_actions):
            for o0 in range(self.task.n_states):
                q_abstract[aa0] += self.task.current_trial.transition_function[s, aa0, o0] * (
                    self.reward_function[c, o0] + self.gamma*v[o0])

        return q_abstract

    def get_mapping(self, state):
        return np.squeeze(self.mapping_mle[state[1], :, :])

    def select_abstract_action(self, state):
        q_abstract = self.get_abstract_q_values(state)
        pmf = softmax_to_pdf(q_abstract, self.inverse_temperature) # doesn't work!!!
        return sample_cmf(pmf.cumsum())

    def get_optimal_policy(self, state):
        _, c = state

        q_abstract = self.get_abstract_q_values(state)
        mapping = self.get_mapping(state)

        return make_q_primitive(q_abstract, mapping)


class JointClustering(FlatAgentQValues):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0,  discount_rate=0.8, iteration_criterion=0.01,
                 mapping_prior=0.01, epsilon=0.025):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.inverse_temperature = inverse_temperature
        # inverse temperature is used internally by the reward hypothesis to convert q-values into a PMF. We
        # always want a very greedy PMF as this is only used to deal with cases where there are multiple optimal
        # actions
        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions
        self.epsilon = epsilon

        # get the list of enumerated set assignments!
        set_assignments = enumerate_assignments(self.task.n_ctx)

        # create task sets, each containing a reward and mapping hypothesis with the same assignement
        self.task_sets = []
        for assignment in set_assignments:

            self.task_sets.append({
                'Reward Hypothesis': RewardHypothesis(
                    self.task.n_states, inverse_temperature, discount_rate, iteration_criterion,
                    assignment, alpha
                ),
                'Mapping Hypothesis': MappingHypothesis(
                    self.task.n_primitive_actions, self.task.n_abstract_actions, assignment, alpha, mapping_prior
                ),
            })

        self.belief = np.ones(len(self.task_sets)) / float(len(self.task_sets))

    def updating_mapping(self, c, a, aa):
        for ts in self.task_sets:
            h_m = ts['Mapping Hypothesis']
            assert type(h_m) is MappingHypothesis
            h_m.updating_mapping(c, a, aa)

    def update_rewards(self, c, sp, r):
        for ts in self.task_sets:
            h_r = ts['Reward Hypothesis']
            assert type(h_r) is RewardHypothesis
            h_r.update(c, sp, r)

    def update(self, experience_tuple):

        # super(FlatAgent, self).update(experience_tuple)
        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

        # then update the posterior
        belief = np.zeros(len(self.task_sets))
        for ii, ts in enumerate(self.task_sets):
            h_m = ts['Mapping Hypothesis']
            h_r = ts['Reward Hypothesis']

            assert type(h_m) is MappingHypothesis
            assert type(h_r) is RewardHypothesis

            log_posterior = h_m.get_log_prior() + h_m.get_log_likelihood() + h_r.get_log_likelihood()
            belief[ii] = np.exp(log_posterior)

        # normalize the posterior
        belief /= np.sum(belief)

        self.belief = belief

    def select_abstract_action(self, state):
        # use epsilon greedy choice function:
        if np.random.rand() > self.epsilon:
            (x, y), c = state
            s = self.task.state_location_key[(x, y)]

            q_values = np.zeros(self.n_abstract_actions)
            for ii, ts in enumerate(self.task_sets):
                # need the posterior (which is calculated during the update) and the pmf from the reward function
                h_r = ts['Reward Hypothesis']

                q_values += h_r.select_abstract_action_pmf(s, c, self.task.current_trial.transition_function) * self.belief[ii]

            full_pmf = np.exp(q_values * self.inverse_temperature)
            full_pmf = full_pmf / np.sum(full_pmf)

            return sample_cmf(full_pmf.cumsum())
        else:
            return np.random.randint(self.n_abstract_actions)

    def select_action(self, state):
        # use epsilon greedy choice function
        if np.random.rand() > self.epsilon:
            _, c = state
            aa = self.select_abstract_action(state)
            c = np.int32(c)

            # print "context:", c, "abstract action:", aa
            mapping_mle = np.zeros(self.n_primitive_actions)
            for ii, ts in enumerate(self.task_sets):
                h_m = ts['Mapping Hypothesis']

                _mapping_mle = np.zeros(self.n_primitive_actions)
                for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
                    # print h_m.get_mapping_probability(c, a0, aa)
                    _mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

                # print ii, self.belief[ii], _mapping_mle
                mapping_mle += _mapping_mle * self.belief[ii]

            return sample_cmf(mapping_mle.cumsum())
        else:
            return np.random.randint(self.n_primitive_actions)

    def get_abstract_q_values(self, state):
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]
        ts = self.task_sets[np.argmax(self.belief)]
        h_r = ts['Reward Hypothesis']

        q_abstract = h_r.get_abstract_action_q_values(s, c, self.task.current_trial.transition_function)

        return q_abstract

    def get_mapping(self, state):
        _, c = state
        ts = self.task_sets[np.argmax(self.belief)]
        h_m = ts['Mapping Hypothesis']

        mapping_prob = np.zeros((self.task.n_primitive_actions, self.task.n_abstract_actions))
        for a in range(self.task.n_primitive_actions):
            for aa in range(self.task.n_abstract_actions):
                mapping_prob[a, aa] = h_m.get_pr_a_given_aa(c, a, aa)

        return mapping_prob

    def set_reward_prior(self, list_locations):
        """
        This method allows the agent to specific grid coordinates as potential goal locations by
        putting some prior (low confidence) reward density over the grid locations.

        All other locations have low reward probability

        :param list_locations: a list of (x, y) coordinates to consider as priors for the goal location search
        :return: None
        """
        list_states = list()
        for loc in list_locations:
            list_states.append(self.task.state_location_key[loc])

        for ii in range(len(self.belief)):
            ts = self.task_sets[ii]
            h_r = ts['Reward Hypothesis']
            h_r.set_reward_prior(list_states)

    def get_map_conditional_goal_probability(self, context, mapping, goals):
        """

        :param context:
        :param mapping: a relationship between primitives and abstract actions, dictionary
        :param goals: a dictionary (?) of goal assignments? (goal # --> state #)
        :return:
        """

        # loop through task sets to get posterior conditional on mapping
        posterior = self.belief.copy()
        for ii, ts in enumerate(self.task_sets):
            h_m = ts['Mapping Hypothesis']
            assert type(h_m) is MappingHypothesis

            mapping_likelihood = 0
            for a, aa in mapping.iteritems():
                mapping_likelihood += np.log(h_m.get_pr_a_given_aa(context, a, aa))

            posterior[ii] *= np.exp(mapping_likelihood)

        posterior /= posterior.sum()

        # loop through and get value function (ignore walls, distance, movement noise, etc)
        n_goals = len(goals)
        goal_probability = np.zeros(n_goals)
        for ii, ts in enumerate(self.task_sets):
            h_r = ts['Reward Hypothesis']
            assert type(h_r) is RewardHypothesis
            reward_visits = h_r.get_reward_visits(context)

            # pull only the goal locations and normalize
            reward_visits = [reward_visits[goals[g]] for g in range(n_goals)]
            reward_visits /= np.sum(reward_visits)

            # collect the goal probabilities
            for g in range(n_goals):
                goal_probability[g] += reward_visits[g] * posterior[ii]

        return goal_probability


        # with the goal probability for each hypothesis, should be able to analytically calculate
        # the probability of the observed sequence conditional on each hypothesis, and weigh
        # that probability with the posterior to get a posterior probibility of the sequence! conditional
        # on alpha

    def evaluate_goal_guess_sequence_probability(self, subject_data, goal_key=None):

        # make the goal key
        if goal_key is None:
            goal_key = {(0, 0): 0, (0, 2): 1, (2, 0): 2, (2, 2): 3}

        goals = np.zeros(len(goal_key.values()), dtype=int)

        for loc, g in goal_key.iteritems():
            goals[g] = self.task.trials[0].state_location_key[loc]

        def make_map(current_trial):
            map_ = dict()
            for a, dir_ in current_trial.action_map.iteritems():
                map_[a] = current_trial.abstract_action_key[dir_]
            return map_

        new_trial = True
        goal_pmf_list = list()
        for exp in subject_data.experience:
            (_, c), _, _, r, _ = exp
            if new_trial:
                map_ = make_map(self.task.current_trial)
                goal_pmf_list.append(self.get_map_conditional_goal_probability(c, map_, goals))
            self.update(exp)
            new_trial = r == 1

        # get the goal guess sequence
        goal_guess_sequence = get_goal_guess_sequence(subject_data)

        # get the likelihood!
        ll = 0
        for ii, guess_sequence in enumerate(goal_guess_sequence):
            goal_pmf = goal_pmf_list[ii]

            for guess in guess_sequence:
                ll += np.log(goal_pmf[guess])

                # renormalize the guess sequence
                goal_pmf[guess] = 0

                goal_pmf /= np.sum(goal_pmf)

        return -ll  # returns negative log likelihood


class IndependentClusterAgent(FlatAgentQValues):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0, discount_rate=0.8, iteration_criterion=0.01,
                 mapping_prior=0.01, epsilon=0.025):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.inverse_temperature = inverse_temperature
        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions
        self.epsilon = epsilon

        # get the list of enumerated set assignments!
        set_assignments = enumerate_assignments(self.task.n_ctx)

        # create task sets, each containing a reward and mapping hypothesis with the same assignement
        self.reward_hypotheses = []
        self.mapping_hypotheses = []
        for assignment in set_assignments:

            self.reward_hypotheses.append(
                RewardHypothesis(
                    self.task.n_states, inverse_temperature, discount_rate, iteration_criterion,
                    assignment, alpha
                )
            )

            self.mapping_hypotheses.append(
                MappingHypothesis(
                    self.task.n_primitive_actions, self.task.n_abstract_actions, assignment, alpha, mapping_prior
                ),
            )

        self.belief_rew = np.ones(len(self.reward_hypotheses)) / float(len(self.reward_hypotheses))
        self.belief_map = np.ones(len(self.mapping_hypotheses)) / float(len(self.mapping_hypotheses))

    def updating_mapping(self, c, a, aa):
        for h_m in self.mapping_hypotheses:
            assert type(h_m) is MappingHypothesis
            h_m.updating_mapping(c, a, aa)

    def update_rewards(self, c, sp, r):
        for h_r in self.reward_hypotheses:
            assert type(h_r) is RewardHypothesis
            h_r.update(c, sp, r)

    def update(self, experience_tuple):

        # super(FlatAgent, self).update(experience_tuple)
        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

        # then update the posterior of the rewards
        belief = np.zeros(len(self.reward_hypotheses))
        for ii, h_r in enumerate(self.reward_hypotheses):
            assert type(h_r) is RewardHypothesis
            log_posterior = h_r.get_log_posterior()
            belief[ii] = np.exp(log_posterior)

        # normalize the posterior
        belief /= np.sum(belief)

        self.belief_rew = belief

        # then update the posterior of the mappings
        belief = np.zeros(len(self.mapping_hypotheses))
        for ii, h_m in enumerate(self.mapping_hypotheses):
            assert type(h_m) is MappingHypothesis
            log_posterior = h_m.get_log_posterior()
            belief[ii] = np.exp(log_posterior)

        # normalize the posterior
        belief /= np.sum(belief)

        self.belief_map = belief

    def select_abstract_action(self, state):
        # use epsilon greedy choice function
        if np.random.rand() > self.epsilon:
            (x, y), c = state
            s = self.task.state_location_key[(x, y)]

            q_values = np.zeros(self.n_abstract_actions)
            for ii, h_r in enumerate(self.reward_hypotheses):
                # need the posterior (which is calculated during the update) and the pmf from the reward function
                assert type(h_r) is RewardHypothesis
                q_values += h_r.select_abstract_action_pmf(s, c, self.task.current_trial.transition_function) * \
                            self.belief_rew[ii]

            full_pmf = np.exp(q_values * self.inverse_temperature)
            full_pmf = full_pmf / np.sum(full_pmf)

            return sample_cmf(full_pmf.cumsum())
        else:
            return np.random.randint(self.n_abstract_actions)

    def select_action(self, state):
        # use epsilon greedy choice function
        if np.random.rand() > self.epsilon:
            _, c = state
            aa = self.select_abstract_action(state)
            c = np.int32(c)

            # print "context:", c, "abstract action:", aa
            mapping_mle = np.zeros(self.n_primitive_actions)
            for ii, h_m in enumerate(self.mapping_hypotheses):
                assert type(h_m) is MappingHypothesis

                _mapping_mle = np.zeros(self.n_primitive_actions)
                for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
                    # print h_m.get_mapping_probability(c, a0, aa)
                    _mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

                mapping_mle += _mapping_mle * self.belief_map[ii]

            return sample_cmf(mapping_mle.cumsum())
        else:
            return np.random.randint(self.n_primitive_actions)

    def get_abstract_q_values(self, state):

        (x, y), c = state
        s = self.task.state_location_key[(x, y)]
        h_r = self.reward_hypotheses[np.argmax(self.belief_rew)]
        _q_abstract = h_r.get_abstract_action_q_values(s, c, self.task.current_trial.transition_function)

        return _q_abstract

    def get_mapping(self, state):
        _, c = state

        h_m = self.mapping_hypotheses[np.argmax(self.belief_map)]
        mapping_prob = np.zeros((self.task.n_primitive_actions, self.task.n_abstract_actions))
        for a in range(self.task.n_primitive_actions):
            for aa in range(self.task.n_abstract_actions):
                mapping_prob[a, aa] = h_m.get_pr_a_given_aa(c, a, aa)

        return mapping_prob

    def set_reward_prior(self, list_locations):
        """
        This method allows the agent to specific grid coordinates as potential goal locations by
        putting some prior (low confidence) reward density over the grid locations.

        All other locations have low reward probability

        :param list_locations: a list of (x, y) coordinates to consider as priors for the goal location search
        :return: None
        """
        list_states = list()
        for loc in list_locations:
            list_states.append(self.task.state_location_key[loc])

        for ii in range(len(self.reward_hypotheses)):
            h_r = self.reward_hypotheses[ii]
            h_r.set_reward_prior(list_states)

    def get_goal_probability(self, context, goals):
        """

        :param context:
        :param mapping: a relationship between primitives and abstract actions, dictionary
        :param goals: a dictionary (?) of goal assignments? (goal # --> state #)
        :return:
        """

        # loop through task sets to get posterior conditional on mapping
        posterior = self.belief_rew.copy()

        # loop through and get value function (ignore walls, distance, movement noise, etc)
        n_goals = len(goals)
        goal_probability = np.zeros(n_goals)
        for ii, h_r in enumerate(self.reward_hypotheses):
            assert type(h_r) is RewardHypothesis
            reward_visits = h_r.get_reward_visits(context)

            # pull only the goal locations and normalize
            reward_visits = [reward_visits[goals[g]] for g in range(n_goals)]
            reward_visits /= np.sum(reward_visits)

            # collect the goal probabilities
            for g in range(n_goals):
                goal_probability[g] += reward_visits[g] * posterior[ii]

        return goal_probability

    def evaluate_goal_guess_sequence_probability(self, subject_data, goal_key=None):

        # make the goal key
        if goal_key is None:
            goal_key = {(0, 0): 0, (0, 2): 1, (2, 0): 2, (2, 2): 3}

        goals = np.zeros(len(goal_key.values()), dtype=int)

        for loc, g in goal_key.iteritems():
            goals[g] = self.task.trials[0].state_location_key[loc]

        new_trial = True
        goal_pmf_list = list()
        for exp in subject_data.experience:
            (_, c), _, _, r, _ = exp
            if new_trial:
                goal_pmf_list.append(self.get_goal_probability(c, goals))
            self.update(exp)
            new_trial = r == 1

        # get the goal guess sequence
        goal_guess_sequence = get_goal_guess_sequence(subject_data)

        # get the likelihood!
        ll = 0
        for ii, guess_sequence in enumerate(goal_guess_sequence):
            goal_pmf = goal_pmf_list[ii]
            for guess in guess_sequence:
                ll += np.log(goal_pmf[guess])

                # renormalize the guess sequence
                goal_pmf[guess] = 0

                goal_pmf /= np.sum(goal_pmf)

        return -ll  # returns negative log likelihood




