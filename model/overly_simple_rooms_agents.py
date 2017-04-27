import numpy as np
from tqdm import tqdm


def softmax_to_pdf(q_values, inverse_temperature):
    pdf = np.exp(np.array(q_values) * float(inverse_temperature))
    pdf = pdf / np.sum(pdf)
    return pdf


def sample_cmf(cmf):
    return int(np.sum(np.random.rand() > cmf))


class SimpleFlat(object):
    def __init__(self, beta, n_doors=2, r_set=None):
        self.r_events = []
        self.beta = beta

        self.n_doors = n_doors
        self.visited_rooms = set()

        if r_set is None:
            r_set = {-1.0, 0, 1.0}
        self.r_set = r_set

    def pick_door(self, room):

        # if this is a new room, generate its prior over the clusters
        if room not in self.visited_rooms:
            self.visited_rooms.add(room)

            # initialize new q functions for the new (potential) cluster
            self.r_events += [{door: {r: 1e-5 for r in self.r_set} for door in range(self.n_doors)}]

        # ~~~~~ have the model pick the goal ~~~~~
        q = np.zeros(self.n_doors)
        for door in range(self.n_doors):
            # get the E[V] of the door!
            n = np.sum(self.r_events[room][door].values())
            for r0, n_r0 in self.r_events[room][door].iteritems():
                q[door] += r0 * n_r0 / n

        pmf = softmax_to_pdf(q, self.beta)
        return sample_cmf(pmf.cumsum())

    def update(self, room, door, r):
        self.r_events[room][door][r] += 1.0


class SimpleIndependent(object):
    def __init__(self, beta, alpha, n_doors=3, r_set=None):

        # initialize the rewards such that each cluster has a prior corresponding to
        # 1e-5 of an event for each possible reward value for each door
        self.r_events = []
        self.beta = beta
        self.alpha = float(alpha)
        self.visited_rooms = set()
        self.cluster_probabilities = dict()

        self.n_doors = n_doors
        if r_set is None:
            r_set = {-1.0, 0, 1.0}
        self.r_set = r_set

    def pick_door(self, room):

        # if this is a new room, generate its prior over the clusters
        if room not in self.visited_rooms:
            self.visited_rooms.add(room)

            n = len(self.visited_rooms) - 1  # total number of rooms visited

            # count the cluster assignments
            cluster_counts = np.zeros(n, dtype=int)
            for room0, pmf in self.cluster_probabilities.iteritems():
                cluster_counts[np.argmax(pmf)] += 1

                # while we're here, augment the size of the previously seen PMFs
                # p(room) = cluster with to be the correct shape
                self.cluster_probabilities[room0] = np.concatenate([pmf, np.zeros(1)])

            # initialize the prior probability p(new room) = cluster with the CRP
            pmf = np.zeros(n + 1)
            for k, n_k in enumerate(cluster_counts):
                pmf[k] = n_k / (self.alpha + n)
            pmf[n] = self.alpha / (self.alpha + n)
            # this is the prior for the new room only!
            self.cluster_probabilities[room] = pmf

            # initialize new q functions for the new (potential) cluster
            self.r_events += [{door: {r:1e-5 for r in self.r_set} for door in range(self.n_doors)}]

        # ~~~~~ choose the maximum a posteriori context assignment ~~~~~
        k_max = np.argmax(self.cluster_probabilities[room])

        # ~~~~~ have the model pick the goal ~~~~~
        q = np.zeros(self.n_doors)
        for door in range(self.n_doors):
            # get the E[V] of the door!
            n = np.sum(self.r_events[k_max][door].values())
            for r0, n_r0 in self.r_events[k_max][door].iteritems():
                q[door] += r0 * n_r0 / n

        pmf = softmax_to_pdf(q, self.beta)
        return sample_cmf(pmf.cumsum())

    def update(self, room, door, r):
        # note! the assumption is that the room will always be in the set of
        # visited rooms, because that update happens when the agent selects an action

        prior = self.cluster_probabilities[room]
        posterior = np.zeros(np.shape(prior)) # initialize
        for k, pr_k in enumerate(prior):

            # Have to regenerate the probability of reward conditional on cluster k
            n = np.sum(self.r_events[k][door].values())
            n_r = self.r_events[k][door][r]
            pr_r = n_r / n

            # update the posterior
            posterior[k] = pr_r * pr_k

        posterior /= posterior.sum()  # normalize
        self.cluster_probabilities[door] = posterior

        k_max = np.argmax(self.cluster_probabilities)

        # update the statics on the MAP cluster only
        self.r_events[k_max][door][r] += 1.0


class SimpleJoint(object):
    def __init__(self, beta, alpha, transitions, n_doors=3, r_set=None):

        # initialize the rewards such that each cluster has a prior corresponding to
        # 1e-5 of an event for each possible reward value for each door
        self.r_events = [[] for _ in set(transitions)]
        self.beta = beta
        self.alpha = float(alpha)
        self.visited_rooms = [set() for _ in set(transitions)]
        self.cluster_probabilities = [dict()  for _ in set(transitions)]

        self.transitions = transitions
        self.n_doors = n_doors

        if r_set is None:
            r_set = {-1.0, 0, 1.0}
        self.r_set = r_set

    def pick_door(self, room):

        t = self.transitions[room]

        # if this is a new room, generate its prior over the clusters
        if room not in self.visited_rooms[t]:
            self.visited_rooms[t].add(room)

            n = len(self.visited_rooms[t]) - 1 # total number of rooms visited

            # count the cluster assignments
            cluster_counts = np.zeros(n, dtype=int)
            for room0, pmf in self.cluster_probabilities[t].iteritems():
                cluster_counts[np.argmax(pmf)] += 1

                # while we're here, augment the size of the previously seen PMFs
                # p(room) = cluster with to be the correct shape
                self.cluster_probabilities[t][room0] = np.concatenate([pmf, np.zeros(1)])

            # initialize the prior probability p(new room) = cluster with the CRP
            pmf = np.zeros(n + 1)
            for k, n_k in enumerate(cluster_counts):
                pmf[k] = n_k / (self.alpha + n)
            pmf[n] = self.alpha / (self.alpha + n)
            # this is the prior for the new room only!
            self.cluster_probabilities[t][room] = pmf

            # initialize new q functions for the new (potential) cluster
            self.r_events[t] += [{door: {r:1e-5 for r in self.r_set} for door in range(self.n_doors)}]

        # ~~~~~ choose the maximum a posteriori context assignment ~~~~~
        k_max = np.argmax(self.cluster_probabilities[t][room])

        # ~~~~~ have the model pick the goal ~~~~~
        q = np.zeros(self.n_doors)
        for door in range(self.n_doors):
            # get the E[V] of the door!
            n = np.sum(self.r_events[t][k_max][door].values())
            for r0, n_r0 in self.r_events[t][k_max][door].iteritems():
                q[door] += r0 * n_r0 / n

        pmf = softmax_to_pdf(q, self.beta)
        return sample_cmf(pmf.cumsum())

    def update(self, room, door, r):
        # note! the assumption is that the room will always be in the set of
        # visited rooms, because that update happens when the agent selects an action
        t = self.transitions[room]
        prior = self.cluster_probabilities[t][room]
        posterior = np.zeros(np.shape(prior)) # initialize
        for k, pr_k in enumerate(prior):

            # Have to regenerate the probability of reward conditional on cluster k
            n = np.sum(self.r_events[t][k][door].values())
            n_r = self.r_events[t][k][door][r]
            pr_r = n_r / n

            # update the posterior
            posterior[k] = pr_r * pr_k

        posterior /= posterior.sum()  # normalize
        self.cluster_probabilities[t][door] = posterior

        k_max = np.argmax(self.cluster_probabilities[t])

        # update the statics on the MAP cluster only
        self.r_events[t][k_max][door][r] += 1.0


def make_room_runner(n_sim, reward_function, sucessor_function):
    def run_rooms(agent_class, params, desc):
        steps_to_goal = [None] * n_sim
        for ii in tqdm(range(n_sim), desc=desc, leave=False):

            current_room = 0
            t = 0

            agent = agent_class(*params)
            while True:
                door = agent.pick_door(current_room)
                r = reward_function[current_room, door]
                new_room = sucessor_function[current_room, door]

                agent.update(current_room, door, r)

                current_room = new_room
                t += 1
                if r == 1:
                    break
            steps_to_goal[ii] = t
        return steps_to_goal
    return run_rooms
