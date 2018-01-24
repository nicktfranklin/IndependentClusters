import numpy as np
import pandas as pd

from gridworld import Task
from cython_library import RewardHypothesis, MappingHypothesis, TransitionHypothesis
from cython_library import policy_iteration, value_iteration


from sklearn.metrics import euclidean_distances
from scipy.misc import logsumexp

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
    :return: list of lists, each a function that takes in a context id number and returns a cluster id number
    """
    cluster_assignments = [{}]  # context 0 is always in cluster 1

    for contextNumber in range(0, max_context_number):
        cluster_assignments = augment_assignments(cluster_assignments, contextNumber)

    return cluster_assignments


def augment_assignments(cluster_assignments, new_context):
    if (len(cluster_assignments) == 0) | (len(cluster_assignments[0]) == 0):
        _cluster_assignments = list()
        _cluster_assignments.append({new_context: 0})
    else:
        _cluster_assignments = list()
        for assignment in cluster_assignments:
            new_list = list()
            for k in range(0, max(assignment.values()) + 2):
                _assignment_copy = assignment.copy()
                _assignment_copy[new_context] = k
                new_list.append(_assignment_copy)
            _cluster_assignments += new_list

    return _cluster_assignments


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


def kl_divergence(q, p):
    d = 0
    for q_ii, p_ii in zip(q, p):
        if p_ii > 0:
            d += p_ii * np.log2(p_ii/q_ii)
    return d


def thompson_sample(pmf):
    # create a cdf from an (un-normalized) pmf
    cmf = np.cumsum(pmf / np.sum(pmf))

    # use inverse CMF sampling
    return np.sum(cmf < np.random.uniform(0, 1.0))




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

    def prune_hypothesis_space(self, threshold=50.):
        pass

    def augment_assignments(self, context):
        pass

    def get_rewards_kl(self, state):
        f_rew = self.get_reward_function(state)
        f_rew /= np.sum(f_rew)
        return kl_divergence(f_rew, self.task.get_reward_function())

    def get_transition_error(self, state):
        pass

    def evaluate_map_rewards(self, state):
        return None, None, None

    def evaluate_map_mapping(self, state):
        pass

    def evaluate_mixing_agent(self, xp, yp, c, r):
        pass

    def is_meta(self):
        return False

    def get_joint_probability(self):
        pass

    def generate(self, pruning_threshold=1000, evaluate=False, evaluate_map_estimate=False):
        """ run through all of the trials of a task and output trial-by-trial data
        :param pruning_threshold:
        :param evaluate: get the KL-divergence of the Model from the task for mappings and rewards
        :param evaluate_map_estimate: evaluate the MAP estimate against the full distribution
        :return:
        """

        # count the number of steps to completion
        step_counter = np.zeros(self.task.n_trials)
        results = list()
        times_seen_ctx = np.zeros(self.task.n_ctx)
        steps_in_ctx = np.zeros(self.task.n_ctx)

        if evaluate_map_estimate:
            pruning_threshold = None

        ii = 0
        while True:

            # get the current state and evaluate stop condition
            state = self.task.get_state()
            if state is None:
                break

            t = self.task.current_trial_number
            step_counter[t] += 1

            _, c = state
            steps_in_ctx[c] += 1

            if step_counter[t] == 1:
                times_seen_ctx[c] += 1

                # entering a new context, prune the hypothesis space and then augment for new context
                if times_seen_ctx[c] == 1:

                    self.prune_hypothesis_space(threshold=pruning_threshold)

                    # augment the clustering assignments
                    self.augment_assignments(c)


            if evaluate_map_estimate:
                kl_map = self.evaluate_map_mapping(state)
                kl_rew, map_rewards, full_rewards = self.evaluate_map_rewards(state)

            if evaluate:
                kl_rew = self.get_rewards_kl(state)
                kl_map = self.get_transition_error(state)
                # kl_rew = 0

            if evaluate & evaluate_map_estimate:
                print "Can't evaluate both map approximation and the estimate!"
                raise Exception

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

            self.evaluate_mixing_agent(xp, yp, c, r)

            # update the learner
            self.update(experience_tuple)

            trial_dict = {
                'Start Location': [(x, y)],
                'End Location': [(xp, yp)],
                'context': [c],
                'key-press': [action],
                'action': [inverse_abstract_action_key[aa]],  # the cardinal movement, in words
                'Reward Collected': [r],
                'n actions taken': step_counter[t],
                'Trial Number': [t],
                'In goal': not (self.task.current_trial_number == t),
                'Times Seen Context': times_seen_ctx[c],
                'action_map': [action_map],
                'goal location': [goal_location],
                'walls': [walls],
                'Steps in Context': steps_in_ctx[c]
            }

            if evaluate:
                trial_dict['KL Rewards'] = kl_rew
                trial_dict['KL Mapping'] = kl_map

            if evaluate_map_estimate:
                trial_dict['KL MAP Mapping'] = kl_map
                trial_dict['MAP Q-function'] = [map_rewards]
                trial_dict['Full Q-Function'] = [full_rewards]
                trial_dict['KL MAP Rewards'] = kl_rew

            if self.is_meta():
                trial_dict['Joint Agent Prob'] = self.get_joint_probability()

            results.append(pd.DataFrame(trial_dict, index=[ii]))

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

        # use the actual action_mapping to get the correct primitive action key
        inverse_abstract_action_key = {aa: move for move, aa in self.task.current_trial.abstract_action_key.iteritems()}
        inverse_action_map = {move: key_press for key_press, move in self.task.current_trial.action_map.iteritems()}

        move = inverse_abstract_action_key[abstract_action]
        key_press = inverse_action_map[move]
        return key_press
        # a = key_press_to_primitive[key_press]


class ModelBasedAgent(FullInformationAgent):
    """ This Agent learns the reward function and mapping will model based planning
    """

    def __init__(self, task, discount_rate=0.8, iteration_criterion=0.01, mapping_prior=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions

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


class JointClustering(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0,  discount_rate=0.8, iteration_criterion=0.01,
                 mapping_prior=0.01):

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

        # create task sets, each containing a reward and mapping hypothesis
        # with the same assignment
        self.reward_hypotheses = [RewardHypothesis(
                self.task.n_states, inverse_temperature, discount_rate, iteration_criterion, alpha
            )]
        self.mapping_hypotheses = [MappingHypothesis(
                self.task.n_primitive_actions, self.task.n_abstract_actions, alpha, mapping_prior
            )]

        self.log_belief = np.ones(1, dtype=float)

    def updating_mapping(self, c, a, aa):
        for h_m in self.mapping_hypotheses:
            assert type(h_m) is MappingHypothesis
            h_m.updating_mapping(c, a, aa)

    def update_rewards(self, c, sp, r):
        for h_r in self.reward_hypotheses:
            assert type(h_r) is RewardHypothesis
            h_r.update(c, sp, r)

    def update(self, experience_tuple):

        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

        self.log_belief = np.zeros(len(self.mapping_hypotheses))
        for ii, h_m in enumerate(self.mapping_hypotheses):
            self.log_belief[ii] = h_m.get_log_prior()

        # then update the posterior of the belief distribution with the reward posterior
        for ii, h_r in enumerate(self.reward_hypotheses):
            assert type(h_r) is RewardHypothesis
            self.log_belief[ii] += h_r.get_log_likelihood()

        # then update the posterior of the mappings likelihood (prior is shared, only need it once)
        for ii, h_m in enumerate(self.mapping_hypotheses):
            assert type(h_m) is MappingHypothesis
            self.log_belief[ii] += h_m.get_log_likelihood()

    def augment_assignments(self, context):
        new_reward_hypotheses = list()
        new_mapping_hypotheses = list()
        new_log_belief = list()

        for h_r, h_m in zip(self.reward_hypotheses, self.mapping_hypotheses):
            assert type(h_r) is RewardHypothesis
            assert type(h_m) is MappingHypothesis

            old_assignments = h_r.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_r0 = h_r.deep_copy()
                h_r0.add_new_context_assignment(context, k)

                h_m0 = h_m.deep_copy()
                h_m0.add_new_context_assignment(context, k)

                new_reward_hypotheses.append(h_r0)
                new_mapping_hypotheses.append(h_m0)
                new_log_belief.append(h_r0.get_log_posterior() + h_m0.get_log_likelihood())

        self.reward_hypotheses = new_reward_hypotheses
        self.mapping_hypotheses = new_mapping_hypotheses
        self.log_belief = new_log_belief

    def prune_hypothesis_space(self, threshold=50.):
        if threshold is not None:
            new_log_belief = []
            new_reward_hypotheses = []
            new_mapping_hypotheses = []
            max_belief = np.max(self.log_belief)

            log_threshold = np.log(threshold)

            for ii, log_b in enumerate(self.log_belief):
                if max_belief - log_b < log_threshold:
                    new_log_belief.append(log_b)
                    new_reward_hypotheses.append(self.reward_hypotheses[ii])
                    new_mapping_hypotheses.append(self.mapping_hypotheses[ii])

            self.log_belief = new_log_belief
            self.reward_hypotheses = new_reward_hypotheses
            self.mapping_hypotheses = new_mapping_hypotheses

    def select_abstract_action(self, state):
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        ii = np.argmax(self.log_belief)
        h_r = self.reward_hypotheses[ii]
        q_values = h_r.select_abstract_action_pmf(s, c, self.task.current_trial.transition_function)

        full_pmf = np.exp(q_values * self.inverse_temperature)
        full_pmf = full_pmf / np.sum(full_pmf)

        return sample_cmf(full_pmf.cumsum())

    def select_action(self, state):
        # use softmax choice function
        _, c = state
        aa = self.select_abstract_action(state)
        c = np.int32(c)

        ii = np.argmax(self.log_belief)
        h_m = self.mapping_hypotheses[ii]

        mapping_mle = np.zeros(self.n_primitive_actions)
        for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
            mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

        return sample_cmf(mapping_mle.cumsum())

    def get_reward_function(self, state):
        # Get the q-values over abstract actions
        _, c = state

        ii = np.argmax(self.log_belief)
        h_r = self.reward_hypotheses[ii]
        return h_r.get_reward_function(c)

    def get_transition_error(self, state):
        _, c = state
        ii = np.argmax(self.log_belief)
        h = self.mapping_hypotheses[ii]

        t = 0
        q = np.zeros(self.task.n_primitive_actions * self.task.n_abstract_actions)
        p = np.zeros(self.task.n_primitive_actions * self.task.n_abstract_actions)
        for aa in range(self.task.n_abstract_actions):
            p_aa = self.task.get_mapping_function(aa)
            for a in range(self.task.n_primitive_actions):
                q[t] = h.get_mapping_probability(c, a, aa)
                p[t] = p_aa[a]
                t += 1

        return kl_divergence(q, p)

    def get_reward_prediction(self, x, y, c):
        sp = self.task.state_location_key[(x, y)]
        ii = np.argmax(self.log_belief)
        h_r = self.reward_hypotheses[ii]
        return h_r.get_reward_prediction(c, sp)

class IndependentClusterAgent(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0, discount_rate=0.8,
                 iteration_criterion=0.01,
                 mapping_prior=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.inverse_temperature = inverse_temperature
        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions

        # get the list of enumerated set assignments!

        # create task sets, each containing a reward and mapping hypothesis
        # with the same assignment
        self.reward_hypotheses = [
            RewardHypothesis(
                self.task.n_states, inverse_temperature, discount_rate,
                iteration_criterion, alpha
            )]
        self.mapping_hypotheses = [
            MappingHypothesis(
                self.task.n_primitive_actions, self.task.n_abstract_actions,
                alpha, mapping_prior
            )]

        self.log_belief_rew = np.ones(1, dtype=float)
        self.log_belief_map = np.ones(1, dtype=float)

    def updating_mapping(self, c, a, aa):
        for h_m in self.mapping_hypotheses:
            assert type(h_m) is MappingHypothesis
            h_m.updating_mapping(c, a, aa)

    def update_rewards(self, c, sp, r):
        for h_r in self.reward_hypotheses:
            assert type(h_r) is RewardHypothesis
            h_r.update(c, sp, r)

    def update(self, experience_tuple):

        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

        # then update the posterior of the rewards
        for ii, h_r in enumerate(self.reward_hypotheses):
            assert type(h_r) is RewardHypothesis
            self.log_belief_rew[ii] = h_r.get_log_posterior()

        # then update the posterior of the mappings
        for ii, h_m in enumerate(self.mapping_hypotheses):
            assert type(h_m) is MappingHypothesis
            self.log_belief_map[ii] = h_m.get_log_posterior()

    def augment_assignments(self, context):
        new_hypotheses = list()
        new_log_belief = list()

        for h_r in self.reward_hypotheses:
            assert type(h_r) is RewardHypothesis

            old_assignments = h_r.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_r0 = h_r.deep_copy()
                h_r0.add_new_context_assignment(context, k)

                new_hypotheses.append(h_r0)
                new_log_belief.append(h_r0.get_log_prior() + h_r0.get_log_likelihood())

        self.reward_hypotheses = new_hypotheses
        self.log_belief_rew = new_log_belief

        new_hypotheses = list()
        new_log_belief = list()

        for h_m in self.mapping_hypotheses:
            assert type(h_m) is MappingHypothesis

            old_assignments = h_m.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_m0 = h_m.deep_copy()
                h_m0.add_new_context_assignment(context, k)

                new_hypotheses.append(h_m0)
                new_log_belief.append(h_m0.get_log_prior() + h_m0.get_log_likelihood())

        self.mapping_hypotheses = new_hypotheses
        self.log_belief_map = new_log_belief

    def prune_hypothesis_space(self, threshold=50.):
        if threshold is not None:
            new_log_belief_rew = []
            new_reward_hypotheses = []
            max_belief = np.max(self.log_belief_rew)

            log_threshold = np.log(threshold)

            for ii, log_b in enumerate(self.log_belief_rew):
                if max_belief - log_b < log_threshold:
                    new_log_belief_rew.append(log_b)
                    new_reward_hypotheses.append(self.reward_hypotheses[ii])

            self.log_belief_rew = new_log_belief_rew
            self.reward_hypotheses = new_reward_hypotheses

            new_log_belief_map = []
            new_mapping_hypotheses = []
            max_belief = np.max(self.log_belief_map)
            for ii, log_b in enumerate(self.log_belief_map):
                if max_belief - log_b < log_threshold:
                    new_log_belief_map.append(log_b)
                    new_mapping_hypotheses.append(self.mapping_hypotheses[ii])

            self.log_belief_map = new_log_belief_map
            self.mapping_hypotheses = new_mapping_hypotheses

    def select_abstract_action(self, state):
        # use softmax greedy choice function
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        ii = np.argmax(self.log_belief_rew)
        h_r = self.reward_hypotheses[ii]

        q_values = h_r.select_abstract_action_pmf(
            s, c, self.task.current_trial.transition_function
        )

        full_pmf = np.exp(q_values * self.inverse_temperature)
        full_pmf = full_pmf / np.sum(full_pmf)

        return sample_cmf(full_pmf.cumsum())

    def get_reward_function(self, state):
        _, c = state

        ii = np.argmax(self.log_belief_rew)
        h_r = self.reward_hypotheses[ii]
        return h_r.get_reward_function(c)

    def get_transition_error(self, state):
        _, c = state
        ii = np.argmax(self.log_belief_map)
        h_m = self.mapping_hypotheses[ii]

        t = 0
        q = np.zeros(self.task.n_primitive_actions * self.task.n_abstract_actions)
        p = np.zeros(self.task.n_primitive_actions * self.task.n_abstract_actions)
        for aa in range(self.task.n_abstract_actions):
            p_aa = self.task.get_mapping_function(aa)
            for a in range(self.task.n_primitive_actions):
                q[t] = h_m.get_mapping_probability(c, a, aa)
                p[t] = p_aa[a]
                t += 1

        return kl_divergence(q, p)

    def evaluate_map_rewards(self, state):
        # Get the q-values over abstract actions
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        ii = np.argmax(self.log_belief_rew)
        h_r = self.reward_hypotheses[ii]

        map_q_values = h_r.select_abstract_action_pmf(
            s, c, self.task.current_trial.transition_function
        )

        belief = np.exp(self.log_belief_rew - np.max(self.log_belief_rew))
        belief /= belief

        full_q_values = np.zeros(self.task.n_abstract_actions)
        for ii, p in enumerate(belief):
            h_r = self.reward_hypotheses[ii]
            full_q_values += h_r.select_abstract_action_pmf(
                s, c, self.task.current_trial.transition_function
            ) * p

        # normalize both
        # map_q_values = np.exp(map_q_values * self.inverse_temperature)
        map_q_values /= np.sum(map_q_values)
        # full_q_values = np.exp(full_q_values * self.inverse_temperature)
        full_q_values /= np.sum(full_q_values)

        return kl_divergence(map_q_values, full_q_values), map_q_values, full_q_values

    def select_action(self, state):
        # use softmax greedy choice function
        _, c = state
        aa = self.select_abstract_action(state)
        c = np.int32(c)

        ii = np.argmax(self.log_belief_map)
        h_m = self.mapping_hypotheses[ii]

        mapping_mle = np.zeros(self.n_primitive_actions)
        for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
            mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

        return sample_cmf(mapping_mle.cumsum())

    def evaluate_map_mapping(self, state):
        # Get the q-values over abstract actions
        _, c = state
        aa = self.select_abstract_action(state)
        c = np.int32(c)

        ii = np.argmax(self.log_belief_map)
        h_m = self.mapping_hypotheses[ii]

        map_mapping_mle = np.zeros(self.n_primitive_actions)
        for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
            map_mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

        # get the full posterior
        belief = np.exp(self.log_belief_map - np.max(self.log_belief_map))
        belief /= np.sum(belief)

        full_mapping_mle = np.zeros(self.n_primitive_actions)
        for ii, p in enumerate(belief):
            h_m = self.mapping_hypotheses[ii]
            for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
                full_mapping_mle[a0] += h_m.get_mapping_probability(c, a0, aa) * p

        # normalize both
        map_mapping_mle /= np.sum(map_mapping_mle)
        full_mapping_mle /= np.sum(full_mapping_mle)

        return kl_divergence(map_mapping_mle, full_mapping_mle)

    def get_reward_prediction(self, x, y, c):
        sp = self.task.state_location_key[(x, y)]
        ii = np.argmax(self.log_belief_rew)
        h_r = self.reward_hypotheses[ii]
        return h_r.get_reward_prediction(c, sp)


class IndependentThompson(IndependentClusterAgent):

    """
    Honestly, this just does a terrible job.

    """

    def select_abstract_action(self, state):
        # use thompson sampling to choose an abstract action
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        # exponentiate and normalize.
        b = np.exp(self.log_belief_rew - np.max(self.log_belief_rew))
        b /= b.sum()
        cdf = np.cumsum(b)
        X = np.sum(np.random.rand() < cdf) - 1

        h_r = self.reward_hypotheses[X]

        q_values = h_r.select_abstract_action_pmf(
            s, c, self.task.current_trial.transition_function
        )

        return np.argmax(q_values)
        # full_pmf = np.exp(q_values * self.inverse_temperature)
        # full_pmf = full_pmf / np.sum(full_pmf)
        #
        # return sample_cmf(full_pmf.cumsum())


class MetaAgent(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0, discount_rate=0.8,
                 iteration_criterion=0.01,
                 mapping_prior=0.01, m_biases=[0.0, 0.0]):
        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.independent_agent = IndependentClusterAgent(
            task, inverse_temperature=inverse_temperature,  alpha=alpha, discount_rate=discount_rate,
                 iteration_criterion=iteration_criterion, mapping_prior=mapping_prior
        )
        self.joint_agent = JointClustering(
            task, inverse_temperature=inverse_temperature,  alpha=alpha, discount_rate=discount_rate,
                 iteration_criterion=iteration_criterion, mapping_prior=mapping_prior
        )

        self.responsibilities = {'Ind': m_biases[0], 'Joint': m_biases[0]}
        # self.responsibilities = {'Ind': 0.5 + mix_bias, 'Joint': 0.5 - mix_bias}
        # self.eta = mixing_lrate
        # self.beta = mixing_temp
        self.is_mixture = True

        self.choose_operating_model()

        # self.current_agent = self.independent_agent
        # self.current_agent_name = 'Ind'
        # if np.random.rand() < 0.5:
        #     self.current_agent = self.joint_agent
        #     self.current_agent_name = 'Joint'

    def is_meta(self):
        return True

    def get_joint_probability(self):
        # k = np.sum(np.exp(self.beta * np.array(self.responsibilities.values())))
        # return np.exp(self.beta * self.responsibilities['Joint']) / k
        return np.exp(self.responsibilities['Joint'] - logsumexp(self.responsibilities.values()))


    def choose_operating_model(self):
        if np.random.rand() < self.get_joint_probability():
            self.current_agent = self.joint_agent
            self.current_agent_name = 'Joint'
        else:
            self.current_agent = self.independent_agent
            self.current_agent_name = 'Ind'


    def update(self, experience_tuple):
        self.independent_agent.update(experience_tuple)
        self.joint_agent.update(experience_tuple)

    def new_trial_function(self):
        self.choose_operating_model()

    def augment_assignments(self, context):
        self.independent_agent.augment_assignments(context)
        self.joint_agent.augment_assignments(context)

    def prune_hypothesis_space(self, threshold=50.):
        self.independent_agent.prune_hypothesis_space(threshold)
        self.joint_agent.prune_hypothesis_space(threshold)

    def select_action(self, state):
        return self.current_agent.select_action(state)

    def evaluate_mixing_agent(self, xp, yp, c, r):
        sp = self.task.state_location_key[(xp, yp)]

        # get the reward prediction for the MAP joint and MAP ind hypotheses
        r_hat_i = self.independent_agent.get_reward_prediction(xp, yp, c)
        r_hat_j = self.joint_agent.get_reward_prediction(xp, yp, c)

        # The map estimate is sensitive to underflow error -- this prevents this be assuming the
        # model has some probability it is wrong (here, hard coded as 1/1000) and bounding the
        # models' probability estimates of reward
        r_hat_j = np.max([0.999 * r_hat_j, 0.001])
        r_hat_i = np.max([0.999 * r_hat_i, 0.001])

        # what is the predicted probability of the observed output for each model? Track the log prob
        self.responsibilities['Joint'] += np.log(r * r_hat_j + (1 - r) * (1 - r_hat_j))
        self.responsibilities['Ind']   += np.log(r * r_hat_i + (1 - r) * (1 - r_hat_i))
        # when r==1, returns the probability of reward; when r==0, return the probability of no reward

class RLMetaAgent(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0, discount_rate=0.8,
                 iteration_criterion=0.01,
                 mapping_prior=0.01, m_biases=[0.0, 0.0], mixing_lrate=0.5, mixing_temp=1.0):
        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.independent_agent = IndependentClusterAgent(
            task, inverse_temperature=inverse_temperature,  alpha=alpha, discount_rate=discount_rate,
                 iteration_criterion=iteration_criterion, mapping_prior=mapping_prior
        )
        self.joint_agent = JointClustering(
            task, inverse_temperature=inverse_temperature,  alpha=alpha, discount_rate=discount_rate,
                 iteration_criterion=iteration_criterion, mapping_prior=mapping_prior
        )

        self.responsibilities = {'Ind': 0.5 + m_biases[0], 'Joint': 0.5 + m_biases[0]}
        # self.responsibilities = {'Ind': 0.5 + mix_bias, 'Joint': 0.5 - mix_bias}
        self.eta = mixing_lrate
        self.beta = mixing_temp
        self.is_mixture = True

        self.choose_operating_model()

    def is_meta(self):
        return True

    def get_joint_probability(self):
        k = np.sum(np.exp(self.beta * np.array(self.responsibilities.values())))
        return np.exp(self.beta * self.responsibilities['Joint']) / k

    def choose_operating_model(self):
        if np.random.rand() < self.get_joint_probability():
            self.current_agent = self.joint_agent
            self.current_agent_name = 'Joint'
        else:
            self.current_agent = self.independent_agent
            self.current_agent_name = 'Ind'


    def update(self, experience_tuple):
        self.independent_agent.update(experience_tuple)
        self.joint_agent.update(experience_tuple)

    def new_trial_function(self):
        self.choose_operating_model()

    def augment_assignments(self, context):
        self.independent_agent.augment_assignments(context)
        self.joint_agent.augment_assignments(context)

    def prune_hypothesis_space(self, threshold=50.):
        self.independent_agent.prune_hypothesis_space(threshold)
        self.joint_agent.prune_hypothesis_space(threshold)

    def select_action(self, state):
        return self.current_agent.select_action(state)


    def evaluate_mixing_agent(self, xp, yp, c, r):
        sp = self.task.state_location_key[(xp, yp)]

        # get the reward prediction for the MAP joint and MAP ind hypotheses
        r_hat_i = self.independent_agent.get_reward_prediction(xp, yp, c)
        r_hat_j = self.joint_agent.get_reward_prediction(xp, yp, c)

        self.responsibilities['Ind'] += self.eta*(r - r_hat_i)
        self.responsibilities['Joint'] += self.eta*(r - r_hat_j)

class IndependentClusterAgentFullBayes(IndependentClusterAgent):

    def prune_hypothesis_space(self, threshold=50.):
        pass

    def select_abstract_action(self, state):
        # use softmax greedy choice function
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        belief = np.exp(self.log_belief_rew - np.max(self.log_belief_rew))
        belief /= belief

        q_values = np.zeros(self.n_abstract_actions)
        for ii, p in enumerate(belief):
            h_r = self.reward_hypotheses[ii]
            q_values += h_r.select_abstract_action_pmf(
                s, c, self.task.current_trial.transition_function
            ) * p

        full_pmf = np.exp(q_values * self.inverse_temperature)
        full_pmf = full_pmf / np.sum(full_pmf)

        return sample_cmf(full_pmf.cumsum())

    def select_action(self, state):
        # use softmax greedy choice function
        _, c = state
        aa = self.select_abstract_action(state)
        c = np.int32(c)

        # get the full posterior
        belief = np.exp(self.log_belief_map - np.max(self.log_belief_map))
        belief /= np.sum(belief)

        mapping_mle = np.zeros(self.n_primitive_actions)
        for ii, p in enumerate(belief):
            h_m = self.mapping_hypotheses[ii]
            for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
                mapping_mle[a0] += h_m.get_mapping_probability(c, a0, aa) * p

        mapping_mle /= np.sum(mapping_mle)
        return sample_cmf(mapping_mle.cumsum())


class FlatControlAgent(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0,
                 discount_rate=0.8, iteration_criterion=0.01,
                 mapping_prior=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.inverse_temperature = inverse_temperature
        # inverse temperature is used internally by the reward hypothesis to convert
        # q-values into a PMF. We
        # always want a very greedy PMF as this is only used to deal with cases where
        # there are multiple optimal
        # actions
        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions

        # create task sets, each containing a reward and mapping hypothesis
        # with the same assignment
        self.task_sets = [{
                'Reward Hypothesis': RewardHypothesis(
                    self.task.n_states, inverse_temperature, discount_rate,
                    iteration_criterion, alpha
                ),
                'Mapping Hypothesis': MappingHypothesis(
                    self.task.n_primitive_actions, self.task.n_abstract_actions,
                    alpha, mapping_prior
                ),
            }]

        self.log_belief = np.ones(1)

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
        for ii, ts in enumerate(self.task_sets):
            h_m = ts['Mapping Hypothesis']
            h_r = ts['Reward Hypothesis']

            assert type(h_m) is MappingHypothesis
            assert type(h_r) is RewardHypothesis

            self.log_belief[ii] = h_m.get_log_prior() + \
                h_m.get_log_likelihood() + h_r.get_log_likelihood()

    def augment_assignments(self, context):

        ts = self.task_sets[0]
        h_m = ts['Mapping Hypothesis']
        h_r = ts['Reward Hypothesis']
        assert type(h_m) is MappingHypothesis
        assert type(h_r) is RewardHypothesis

        h_m.add_new_context_assignment(context, context)
        h_r.add_new_context_assignment(context, context)

        self.task_sets = [{'Reward Hypothesis': h_r, 'Mapping Hypothesis': h_m}]
        self.log_belief = [1]

    def select_abstract_action(self, state):
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        ii = np.argmax(self.log_belief)
        h_r = self.task_sets[ii]['Reward Hypothesis']
        q_values = h_r.select_abstract_action_pmf(
            s, c, self.task.current_trial.transition_function
        )

        full_pmf = np.exp(q_values * self.inverse_temperature)
        full_pmf = full_pmf / np.sum(full_pmf)

        return sample_cmf(full_pmf.cumsum())

    def select_action(self, state):

        _, c = state
        aa = self.select_abstract_action(state)
        c = np.int32(c)

        ii = np.argmax(self.log_belief)
        h_m = self.task_sets[ii]['Mapping Hypothesis']

        mapping_mle = np.zeros(self.n_primitive_actions)
        for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
            mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

        return sample_cmf(mapping_mle.cumsum())

    def get_reward_function(self, state):
        # Get the q-values over abstract actions
        _, c = state

        ii = np.argmax(self.log_belief)
        h_r = self.task_sets[ii]['Reward Hypothesis']
        return h_r.get_reward_function(c)

    def get_transition_error(self, state):
        _, c = state
        ii = np.argmax(self.log_belief)
        h = self.task_sets[ii]['Mapping Hypothesis']

        t = 0
        q = np.zeros(self.task.n_primitive_actions * self.task.n_abstract_actions)
        p = np.zeros(self.task.n_primitive_actions * self.task.n_abstract_actions)
        for aa in range(self.task.n_abstract_actions):
            p_aa = self.task.get_mapping_function(aa)
            for a in range(self.task.n_primitive_actions):
                q[t] = h.get_mapping_probability(c, a, aa)
                p[t] = p_aa[a]
                t += 1

        return kl_divergence(q, p)


class MapClusteringAgent(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0, discount_rate=0.8,
                 iteration_criterion=0.01,
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
        map_set_assignments = enumerate_assignments(self.task.n_ctx)
        set_assignments = [{ii: ii for ii in range(task.n_ctx)}]

        # create task sets, each containing a reward and mapping hypothesis with the
        #  same assignment
        self.reward_hypotheses = []
        self.mapping_hypotheses = []
        for assignment in set_assignments:
            self.reward_hypotheses.append(
                RewardHypothesis(
                    self.task.n_states, inverse_temperature, discount_rate,
                    iteration_criterion, assignment, alpha
                )
            )

        for assignment in map_set_assignments:
            self.mapping_hypotheses.append(
                MappingHypothesis(
                    self.task.n_primitive_actions, self.task.n_abstract_actions,
                    assignment, alpha, mapping_prior
                ),
            )

        self.belief_rew = np.ones(len(self.reward_hypotheses)) /\
            float(len(self.reward_hypotheses))

        self.belief_map = np.ones(len(self.mapping_hypotheses)) /\
            float(len(self.mapping_hypotheses))

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
                # need the posterior (which is calculated during the update) and the
                #  pmf from the reward function
                assert type(h_r) is RewardHypothesis
                q_values += h_r.select_abstract_action_pmf(
                    s, c, self.task.current_trial.transition_function) * \
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

            ii = np.argmax(self.belief_map)
            h_m = self.mapping_hypotheses[ii]

            mapping_mle = np.zeros(self.n_primitive_actions)
            for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
                mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

            return sample_cmf(mapping_mle.cumsum())
        else:
            return np.random.randint(self.n_primitive_actions)


class FlatTransitionAgent(FullInformationAgent):
    """ This Agent learns the reward function and transition functions and uses model based planning
    """

    def __init__(self, task, inverse_temperature=100.0,  discount_rate=0.8, iteration_criterion=0.01, prior=0.01,
                 epsilon=0.05):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.inverse_temperature = inverse_temperature
        # inverse temperature is used internally by the reward hypothesis to convert q-values into a PMF. We
        # always want a very greedy PMF as this is only used to deal with cases where there are multiple optimal
        # actions
        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_actions = self.task.n_primitive_actions  # this model does not use abstract actions
        self.epsilon = epsilon

        # create task sets, each containing a reward and mapping hypothesis
        # with the same assignment
        self.reward_hypotheses = [RewardHypothesis(
                self.task.n_states, inverse_temperature, discount_rate, iteration_criterion, 1.0
            )]
        self.transition_hypotheses = [TransitionHypothesis(
                self.task.n_primitive_actions, self.task.n_states, 1.0, prior
            )]

        self.log_belief_transitions = np.ones(1, dtype=float)
        self.log_belief_rewards = np.ones(1, dtype=float)

    def update(self, experience_tuple):
        (loc, _), a, _, r, (loc_prime, c) = experience_tuple
        s = self.task.state_location_key[loc]
        sp = self.task.state_location_key[loc_prime]

        self.update_transitions(c, s, a, sp)
        self.update_rewards(c, sp, r)

    def update_transitions(self, c, s, a, sp):
        for h_t in self.transition_hypotheses:
            assert type(h_t) is TransitionHypothesis
            h_t.update(c, s, a, sp)

    def update_rewards(self, c, sp, r):
        for h_r in self.reward_hypotheses:
            assert type(h_r) is RewardHypothesis
            h_r.update(c, sp, r)

    def select_action(self, state):

        # use epsilon greedy choice function, with thompson sampling over hypotheses (b/c rewards are deterministic!)
        if np.random.rand() > self.epsilon:
            (loc), c = state
            s = self.task.state_location_key[loc]

            # Thompson Sampling! Draw hypothesis for transition and reward functions and estimate a value function
            # from these samples
            pmf_t = np.exp(self.log_belief_transitions - logsumexp(self.log_belief_transitions))
            pmf_r = np.exp(self.log_belief_rewards - logsumexp(self.log_belief_rewards))

            ii = thompson_sample(pmf_t)
            # ii = np.argmax(self.log_belief_transitions)
            h_t = self.transition_hypotheses[ii]
            assert type(h_t) is TransitionHypothesis
            transition_function = np.asarray(h_t.get_transition_function(c))

            ii = thompson_sample(pmf_r)
            # ii = np.argmax(self.log_belief_rewards)
            h_r = self.reward_hypotheses[ii]
            assert type(h_r) is RewardHypothesis
            reward_function = h_r.get_reward_function(c)

            v = value_iteration(transition_function, reward_function,
                                  gamma=self.gamma,
                                  stop_criterion=self.iteration_criterion)

            # use the bellman equation to solve for q
            q = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for sp in range(self.task.n_states):
                    q[a] += transition_function[s, a, sp] * (reward_function[sp] + self.gamma * v[sp])

            # action = np.argmax(q)
            full_pmf = np.exp(q * self.inverse_temperature)
            full_pmf = full_pmf / np.sum(full_pmf)

            return sample_cmf(full_pmf.cumsum())
        else:
            action = np.random.randint(self.n_actions)

        return action

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

    def prune_hypothesis_space(self, threshold=50.):
        pass

    def augment_assignments(self, context):
        # here, we just deterministically assign each context to a new cluster. There is only one hypothesis
        # in this model

        h_r = self.reward_hypotheses[0]
        h_t = self.transition_hypotheses[0]

        assert type(h_r) is RewardHypothesis
        assert type(h_t) is TransitionHypothesis

        h_r.add_new_context_assignment(context, context)
        h_t.add_new_context_assignment(context, context)

        self.reward_hypotheses[0] = h_r
        self.transition_hypotheses[0] = h_t

    def get_transition_error(self, state):
        _, c = state
        h_t = self.transition_hypotheses[0]

        p = self.task.get_transition_function()

        t = 0
        q = np.zeros((self.task.n_states, self.task.n_primitive_actions, self.task.n_states))

        for s in range(self.task.n_states):
            for a in range(self.task.n_primitive_actions):
                q[s, a, :] = h_t.get_transition_probability(c, s, a)

        p = p.flatten()
        q = q.flatten()
        # return kl_divergence(q, p)
        return euclidean_distances(p.reshape(1, -1), q.reshape(1, -1))[0]

    def get_rewards_kl(self, state):
        return np.array([1.0])

class JointTransitionAgent(FlatTransitionAgent):
    """ This Agent learns the reward function and transition functions and uses model based planning
    """
    def __init__(self, task, inverse_temperature=100.0,  discount_rate=0.8, iteration_criterion=0.01, prior=0.01,
                 epsilon=0.01, alpha = 1.0):
        assert type(task) is Task
        super(JointTransitionAgent, self).__init__(task, inverse_temperature=inverse_temperature,
                                                   discount_rate=discount_rate, iteration_criterion=iteration_criterion,
                                                   prior=prior, epsilon=epsilon)

        self.reward_hypotheses = [RewardHypothesis(
                self.task.n_states, inverse_temperature, discount_rate, iteration_criterion, alpha
            )]
        self.transition_hypotheses = [TransitionHypothesis(
                self.task.n_primitive_actions, self.task.n_states, alpha, prior
            )]

    def augment_assignments(self, context):
        new_reward_hypotheses = list()
        new_transition_hypotheses = list()
        new_log_belief = list()

        for h_r, h_t in zip(self.reward_hypotheses, self.transition_hypotheses):
            assert type(h_r) is RewardHypothesis
            assert type(h_t) is TransitionHypothesis

            old_assignments = h_r.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_r0 = h_r.deep_copy()
                h_r0.add_new_context_assignment(context, k)

                h_t0 = h_t.deep_copy()
                h_t0.add_new_context_assignment(context, k)

                new_reward_hypotheses.append(h_r0)
                new_transition_hypotheses.append(h_t0)
                new_log_belief.append(h_r0.get_log_posterior() + h_t0.get_log_likelihood())

        self.reward_hypotheses = new_reward_hypotheses
        self.transition_hypotheses = new_transition_hypotheses

        # for simplicity of programming, create a log belief over the two hypothesis sets that are identical
        self.log_belief_transitions = np.array(new_log_belief)
        self.log_belief_rewards = np.array(new_log_belief)

    def update(self, experience_tuple):

        super(JointTransitionAgent, self).update(experience_tuple)

        # these two functions should be the same!
        self.log_belief_transitions = np.zeros(len(self.transition_hypotheses))
        self.log_belief_rewards = np.zeros(len(self.log_belief_rewards))
        for ii, h_t in enumerate(self.transition_hypotheses):
            assert type(h_t) is TransitionHypothesis
            self.log_belief_transitions[ii] = h_t.get_log_prior()
            self.log_belief_rewards[ii]     = h_t.get_log_prior()

        # then update the posterior of the belief distribution with the reward posterior
        for ii, h_r in enumerate(self.reward_hypotheses):
            assert type(h_r) is RewardHypothesis

            self.log_belief_transitions[ii] += h_r.get_log_likelihood()
            self.log_belief_rewards[ii]     += h_r.get_log_likelihood()

        # then update the posterior of the mappings likelihood (prior is shared, only need it once)
        for ii, h_t in enumerate(self.transition_hypotheses):
            self.log_belief_transitions[ii] += h_t.get_log_likelihood()
            self.log_belief_rewards[ii]     += h_t.get_log_likelihood()

    def prune_hypothesis_space(self, threshold=50.):
        if threshold is not None:
            new_log_belief = []
            new_reward_hypotheses = []
            new_transition_hypotheses = []
            max_belief = np.max(self.log_belief_transitions)  # the two belief functions are the same

            log_threshold = np.log(threshold)

            for ii, log_b in enumerate(self.log_belief_transitions):
                if max_belief - log_b < log_threshold:
                    new_log_belief.append(log_b)
                    new_reward_hypotheses.append(self.reward_hypotheses[ii])
                    new_transition_hypotheses.append(self.transition_hypotheses[ii])

            self.log_belief_transitions = new_log_belief
            self.log_belief_rewards     = new_log_belief

            self.reward_hypotheses     = new_reward_hypotheses
            self.transition_hypotheses = new_transition_hypotheses


class IndependentTransitionAgent(FlatTransitionAgent):
    """ This Agent learns the reward function and transition functions and uses model based planning
    """
    def __init__(self, task, inverse_temperature=100.0,  discount_rate=0.8, iteration_criterion=0.01, prior=0.01,
                 epsilon=0.01, alpha = 1.0):
        assert type(task) is Task
        super(IndependentTransitionAgent, self).__init__(task, inverse_temperature=inverse_temperature,
                                                   discount_rate=discount_rate, iteration_criterion=iteration_criterion,
                                                   prior=prior, epsilon=epsilon)

        self.reward_hypotheses = [RewardHypothesis(
                self.task.n_states, inverse_temperature, discount_rate, iteration_criterion, alpha
            )]
        self.transition_hypotheses = [TransitionHypothesis(
                self.task.n_primitive_actions, self.task.n_states, alpha, prior
            )]
        
    def augment_assignments(self, context):
        new_reward_hypotheses = list()
        new_transition_hypotheses = list()
        new_log_belief_rew = list()
        new_log_belief_trans = list()

        for h_r in self.reward_hypotheses:
            assert type(h_r) is RewardHypothesis

            old_assignments = h_r.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_r0 = h_r.deep_copy()
                h_r0.add_new_context_assignment(context, k)

                new_reward_hypotheses.append(h_r0)
                new_log_belief_rew.append(h_r0.get_log_prior() + h_r0.get_log_likelihood())

        self.reward_hypotheses = new_reward_hypotheses
        self.log_belief_rewards = new_log_belief_rew

        for h_t in self.transition_hypotheses:
            assert type(h_t) is TransitionHypothesis

            old_assignments = h_t.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_t0 = h_t.deep_copy()
                h_t0.add_new_context_assignment(context, k)

                new_transition_hypotheses.append(h_t0)
                new_log_belief_trans.append(h_t0.get_log_prior() + h_t0.get_log_likelihood())

        self.transition_hypotheses = new_transition_hypotheses
        self.log_belief_transitions = new_log_belief_trans

    def update(self, experience_tuple):

        super(IndependentTransitionAgent, self).update(experience_tuple)

        # first, update the transition cluster posterior
        self.log_belief_transitions = np.zeros(len(self.log_belief_transitions))
        for ii, h_t in enumerate(self.transition_hypotheses):
            assert type(h_t) is TransitionHypothesis
            self.log_belief_transitions[ii] = h_t.get_log_posterior()

        # then update the posterior of the belief distribution with the reward posterior
        self.log_belief_rewards = np.zeros(len(self.log_belief_rewards))
        for ii, h_r in enumerate(self.reward_hypotheses):
            assert type(h_r) is RewardHypothesis
            self.log_belief_rewards[ii]     += h_r.get_log_posterior()


    def prune_hypothesis_space(self, threshold=50.):
        if threshold is not None:
            new_log_belief_rew = []
            new_log_belief_transitions = []
            new_reward_hypotheses = []
            new_transition_hypotheses = []
            max_belief_rew = np.max(self.log_belief_rewards)
            max_belief_trans = np.max(self.log_belief_transitions)

            log_threshold = np.log(threshold)

            for ii, log_b in enumerate(self.log_belief_rewards):
                if max_belief_rew - log_b < log_threshold:
                    new_log_belief_rew.append(log_b)
                    new_reward_hypotheses.append(self.reward_hypotheses[ii])

            for ii, log_b in enumerate(self.log_belief_transitions):
                if max_belief_trans - log_b < log_threshold:
                    new_log_belief_transitions.append(log_b)
                    new_transition_hypotheses.append(self.transition_hypotheses[ii])

            self.log_belief_transitions = new_log_belief_transitions
            self.log_belief_rewards     = new_log_belief_rew

            self.reward_hypotheses     = new_reward_hypotheses
            self.transition_hypotheses = new_transition_hypotheses
