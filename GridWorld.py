import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

""" the purpose of this code is to take a subject's trial and create the relevant grid-world for that

Things needed:
1) walls
2) grid world shape
3) veridical action map
4) veridical goal location
5) context number (not really... this is something passed to the agent, but isn't the grid world per se)


# this is an example wall format:
walls = [[0, 0, u'right'], [1, 0, u'left'], [1, 2, u'right'], [2, 2, u'left']]

# its safe to assume that the grid world is 3x3 (do want these as parameters, though, for future experiments)
gw_size = (3, 3)

# this is an example action map format:
action_map = {u'65': u'left', u'68': u'down', u'70': u'right', u'83': u'up'}

# goal location
goal = (0, 0)


"""


# code to make the grid world starts here!


class GridWorld(object):
    def __init__(self, grid_world_size, walls, action_map, goal, start_location, context,
                 state_location_key=None, n_abstract_actions=4):
        """

        :param grid_world_size: 2x2 tuple
        :param walls: list of [x, y, 'direction_of_wall'] lists
        :param action_map: dictionary of from {a: 'cardinal direction'}
        :param goal: tuple (x, y)
        :param start_location: tuple (x, y)
        :param n_abstract_actions: int
        :return:
        """
        self.start_location = start_location
        self.current_location = start_location
        self.goal_location = goal
        self.grid_world_size = grid_world_size
        self.context = int(context)
        self.walls = walls

        # need to create a transition function and reward function, which pretty much define the grid world
        n_states = grid_world_size[0] * grid_world_size[1]  # assume rectangle...
        if state_location_key is None:
            self.state_location_key = {
                (x, y): (y + x * grid_world_size[1]) for y in range(grid_world_size[1]) for x in
                range(grid_world_size[0])
                }
        else:
            self.state_location_key = state_location_key

        self.inverse_state_loc_key = {value: key for key, value in self.state_location_key.iteritems()}

        # define movements as change in x and y position:
        self.cardinal_direction_key = {u'up': (0, 1), u'down': (0, -1), u'left': (-1, 0), u'right': (1, 0)}
        self.abstract_action_key = {dir_: ii for ii, dir_ in enumerate(self.cardinal_direction_key.keys())}
        self.abstract_action_key[u'wait'] = -1

        self.inverse_abstract_action_key = {ii: dir_ for dir_, ii in self.abstract_action_key.iteritems()}

        # redefine walls pythonicly:
        wall_key = {(x, y): wall_side for x, y, wall_side in walls}
        wall_list = wall_key.keys()

        # make transition function (usable by the agent)!
        # transition function: takes in state, abstract action, state' and returns probability
        self.transition_function = np.zeros((n_states, n_abstract_actions, n_states), dtype=float)
        for s in range(n_states):

            x, y = self.inverse_state_loc_key[s]

            # cycle through movement, check for both walls and for
            for movement, (dx, dy) in self.cardinal_direction_key.iteritems():
                aa = self.abstract_action_key[movement]

                # check if the movement stays on the grid
                if (x + dx, y + dy) not in self.state_location_key.keys():
                    self.transition_function[s, aa, s] = 1

                elif (x, y) in wall_list:
                    # check if the movement if blocked by a wall
                    if wall_key[(x, y)] == movement:
                        self.transition_function[s, aa, s] = 1
                    else:
                        sp = self.state_location_key[(x + dx, y + dy)]
                        self.transition_function[s, aa, sp] = 1
                else:
                    sp = self.state_location_key[(x + dx, y + dy)]
                    self.transition_function[s, aa, sp] = 1

        # make reward function
        self.reward_function = np.zeros((n_states, n_states), dtype=float)
        self.reward_function[:, :] = -0.1
        self.reward_function[:, self.state_location_key[goal]] = 1.0

        # store the action map
        self.action_map = {int(key): value for key, value in action_map.iteritems()}

        self.n_primitive_actions = len(self.action_map.keys())

        # define a successor function in terms of key-press for game interactions
        # successor function: takes in location (x, y) and action (button press) and returns successor location (x, y)
        self.successor_function = dict()
        for s in range(n_states):
            x, y = self.inverse_state_loc_key[s]

            # this loops through keys associated with a movement (valid key-presses only)
            for key_press, movement in self.action_map.iteritems():
                dx, dy = self.cardinal_direction_key[movement]

                if (x + dx, y + dy) not in self.state_location_key.keys():
                    self.successor_function[((x, y), key_press)] = (x, y)

                # check the walls for valid movements
                elif (x, y) in wall_list:
                    if wall_key[(x, y)] == movement:
                        self.successor_function[((x, y), key_press)] = (x, y)
                    else:
                        self.successor_function[((x, y), key_press)] = (x + dx, y + dy)
                else:
                    self.successor_function[((x, y), key_press)] = (x + dx, y + dy)

        # store keys used in the task for lookup value
        self.keys_used = [key for key in self.action_map.iterkeys()]

        # store walls
        self.wall_key = wall_key

    def move(self, key_press):
        """
        :param key_press: int key-press
        :return: tuple (((x, y), c), a, aa, r, ((xp, yp), c))
        """
        if key_press in self.keys_used:
            new_location = self.successor_function[self.current_location, key_press]
            # get the abstract action number
            aa = self.abstract_action_key[self.action_map[key_press]]
        else:
            new_location = self.current_location
            aa = -1

        # get reward
        r = self.reward_function[
            self.state_location_key[self.current_location], self.state_location_key[new_location]
        ]

        # update the current location before returning
        self.current_location = new_location

        return (self.current_location, self.context), key_press, aa, r, (new_location, self.context)

    def goal_check(self):
        return self.current_location == self.goal_location

    def get_state(self):
        return self.current_location, self.context

    def draw_state(self, fig=None, ax=None):
        x, y = self.current_location

        if ax == None:
            fig, ax = plt.subplots(figsize=(2, 2))

        self._draw_grid(ax=ax)
        self._draw_wall(ax=ax)

        ax.plot([x, x], [y, y], 'bo', markersize=15)

        # draw the goal location:
        ax.annotate('G', xy=self.goal_location, xytext=self.goal_location, size=15)
        return fig, ax

    def draw_move(self, key_press, fig=None, ax=None):
        fig, ax = self.draw_state(fig=fig, ax=ax)
        print "Current State:", self.get_state()

        # use the successor function!
        if key_press in self.keys_used:
            (xp, yp) = self.successor_function[self.current_location, key_press]
            print "Key Press:", str(key_press), "; Corresponding Movement:", self.action_map[key_press]
        else:
            (xp, yp) = self.current_location
            print "Key Press:", str(key_press), "; Corresponding Movement: None!"

        ax.plot([xp, xp], [yp, yp], 'go', markersize=20)

        x0, y0 = self.current_location
        dx = xp - x0
        dy = yp - y0
        dx = (abs(dx) - 0.5) * np.sign(dx)
        dy = (abs(dy) - 0.5) * np.sign(dy)
        ax.arrow(x0, y0, dx, dy, width=0.005, color='k')

        # plt.show()
        time.sleep(0.1)

        return fig, ax

    def _draw_grid(self, ax):
        # first use the gw_size to draw the most basic grid
        sns.set_style('white')
        for x in range(self.grid_world_size[0] + 1):
            for y in range(self.grid_world_size[1] + 1):
                ax.plot([-0.5, self.grid_world_size[0] - 0.5], [y - 0.5, y - 0.5], color=[0.75, 0.75, 0.75])
                ax.plot([x - 0.5, x - 0.5], [-0.5, self.grid_world_size[1] - 0.5], color=[0.75, 0.75, 0.75])

        # finish the plot!
        sns.despine(left=True, bottom=True)
        ax.set_axis_off()

    def _draw_wall(self, ax):
        # plot the walls!
        for (x, y), direction in self.wall_key.iteritems():
            if direction == u'right':
                ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], 'k')
            elif direction == u'left':
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], 'k')
            elif direction == u'up':
                ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], 'k')
            else:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], 'k')

    def draw_transition_function(self):
        # first use the gw_size to draw the most basic grid
        sns.set_style('white')

        fig, ax = plt.subplots(figsize=(4, 4))
        self._draw_grid(ax=ax)

        # maybe draw the actions with different colors?
        action_color_code = {0: 'b', 1: 'r', 2: 'g', 3: 'm'}
        for s in range(self.transition_function.shape[0]):
            for a in range(self.transition_function.shape[1]):
                for sp in range(self.transition_function.shape[2]):
                    if (self.transition_function[s, a, sp] == 1) & (not s == sp):
                        color = action_color_code[a]
                        x, y = self.inverse_state_loc_key[s]
                        xp, yp = self.inverse_state_loc_key[sp]
                        dx = xp - x
                        dy = yp - y

                        # make the change smaller. taking the absolute value and multiplying by its sign
                        # ensures that if there is no change, the derviative is still zero
                        dx = (abs(dx) - 0.75) * np.sign(dx)
                        dy = (abs(dy) - 0.75) * np.sign(dy)
                        ax.arrow(x, y, dx, dy, width=0.005, color=color)

        self._draw_wall(ax=ax)

        # finish the plot!
        plt.show()


class Task(object):
    """ This is a data structure that holds all of the trials a subject encountered in a format readable by the models.
    This is used primarily for the purposes of initialization of the agents.
    """

    def __init__(self, list_start_location, list_end_location, list_context, list_action_map,
                 grid_world_size=(3, 3),
                 n_abstract_actions=4,
                 primitive_actions=(72, 74, 75, 76, 65, 83, 68, 70),
                 list_walls=[],
                 ):
        """

        :param subject_data: pandas.DataFrame (raw from experiment)
        :return: None
        """
        self.grid_world_size = grid_world_size
        self.n_states = grid_world_size[0] * grid_world_size[1]
        self.n_abstract_actions = n_abstract_actions
        self.n_primitive_actions = len(primitive_actions)
        self.primitive_actions = primitive_actions

        # count the number of trials
        self.n_trials = len(list_context)

        # count the number of contexts
        self.n_ctx = len(set(list_context))

        # create a state location key
        self.state_location_key = {
            (x, y): (y + x * grid_world_size[1]) for y in range(grid_world_size[1]) for x in range(grid_world_size[0])
            }
        self.inverse_state_loc_key = {value: key for key, value in self.state_location_key.iteritems()}

        # create a key-code between keyboard presses and action numbers
        self.keyboard_action_code = {unicode(keypress): a for a, keypress in enumerate(primitive_actions)}

        # for each trial, I need the walls, action_map and goal location and start location to define the grid world
        self.trials = list()
        if len(list_walls) == 0:
            list_walls = [None] * self.n_trials

        for ii in range(self.n_trials):

            self.trials.append(
                GridWorld(
                    grid_world_size,
                    list_walls[ii],
                    list_action_map[ii],
                    list_end_location[ii],
                    list_start_location[ii],
                    list_context[ii],
                    state_location_key=self.state_location_key,
                    n_abstract_actions=n_abstract_actions
                )
            )

        # set the current trial
        self.current_trial_number = 0
        self.current_trial = self.trials[0]

    def get_state(self):
        if self.current_trial is not None:
            return self.current_trial.get_location()

    def move(self, action):
        # key_press = self.primitive_actions[self.keyboard_action_code[action]]

        # self.current_trial.draw_move(key_press)
        s, _, aa, r, sp = self.current_trial.move(action)

        if self.current_trial.goal_check():
            self.current_trial_number += 1
            if self.current_trial_number < len(self.trials):
                self.current_trial = self.trials[self.current_trial_number]
            else:
                self.current_trial = None

        return s, action, aa, r, sp


def _prep_sub_data(subject_data, keyboard_action_code, grid_world_size=(3, 3), n_abstract_actions = 4):

    assert type(subject_data) is pd.DataFrame

    # count the number of trials
    n_trials = int(subject_data.TrialNumber.max() + 1)

    #
    primitive_actions = keyboard_action_code.keys()

    # create a list of:
    list_walls = []
    list_start_location = []
    list_end_location = []
    list_context = []
    list_action_map = []

    for ii in range(n_trials):
        trial_data = subject_data[subject_data.TrialNumber == ii]

        # convert the action map to a function of action numbers
        action_map = trial_data[u'action_map'][trial_data.index[0]]
        action_map = {keyboard_action_code[key]: value for key, value in action_map.iteritems()}

        list_walls.append(trial_data[u'walls'][trial_data.index[0]])
        list_start_location.append(tuple(trial_data[u'Start Location'][trial_data.index[0]]))
        list_end_location.append(tuple(trial_data[u'End Location'][trial_data.index[-1]]))
        list_context.append(trial_data[u'context'][trial_data.index[0]] - 1)
        list_action_map.append(action_map)

    kwargs = {
        'grid_world_size': grid_world_size,
        'n_abstract_actions': n_abstract_actions,
        'primitive_actions': primitive_actions,
        'list_walls': list_walls,
        'list_start_location': list_start_location,
        'list_end_location': list_end_location,
        'list_context': list_context,
        'list_action_map': list_action_map
    }
    return kwargs


def make_task_from_subject(subject_data, grid_world_size=(3, 3), n_abstract_actions=4,
                           primitive_actions=(72, 74, 75, 76, 65, 83, 68, 70)):

    # clean the data set (remove non-experiment trials)
    subject_data = subject_data[subject_data.Phase == 'Experiment'].copy()

    # create a key-code between keyboard presses and action numbers
    keyboard_action_code = {unicode(keypress): a for a, keypress in enumerate(primitive_actions)}

    kwargs = _prep_sub_data(subject_data, keyboard_action_code, grid_world_size=grid_world_size,
                            n_abstract_actions=n_abstract_actions)

    return Task(**kwargs)


def make_task_from_generative(subject_data, grid_world_size=(3, 3), n_abstract_actions=4,
                              primitive_actions=(0, 1, 2, 3, 4, 5, 6, 7)):

    keyboard_action_code = {ii: ii for ii in primitive_actions}

    kwargs = _prep_sub_data(subject_data, keyboard_action_code, grid_world_size=grid_world_size,
                            n_abstract_actions=n_abstract_actions)
    return Task(**kwargs)


def get_goal_guess_sequence(data_frame, goal_key=None):

        if goal_key is None:
            goal_key = {(0, 0): 0, (0, 2): 1, (2, 0): 2, (2, 2): 3}

        locations_visited = set()

        guess_sequence = list()
        all_guess_sequence = list()
        for _, p, a, r, (loc, c) in data_frame.experience:

            # goal check (only use the first goals guessed for now)
            if loc in goal_key.keys() and (loc not in locations_visited):
                guess_sequence.append(goal_key[loc])

            locations_visited.update([tuple(loc)])

            if r == 1:
                all_guess_sequence.append(guess_sequence)
                guess_sequence = list()
                locations_visited = set()

        return all_guess_sequence


class SubjectData(object):
    """ this is a data structure that holds all of the subjects trial by trial data, used for fitting etc.
    """

    def __init__(self, subject_data, grid_world_size=(3, 3), n_abstract_actions=4,
                 primitive_actions=(72, 74, 75, 76, 65, 83, 68, 70)):

        self.Task = make_task_from_subject(subject_data, grid_world_size, n_abstract_actions, primitive_actions)
        subject_data = subject_data[subject_data.Phase == 'Experiment'].copy()

        # need the equivalent of SARS tuples
        # inverse_state_loc_key = self.

        # need: start position, end position & keypress for each step
        self.trial_history = list()
        self.experience = list()
        for ii in range(self.Task.n_trials):

            trial_data = subject_data[subject_data.TrialNumber == ii]

            # needed for encoding cardinal directions numerically
            abstract_action_key = self.Task.trials[ii].abstract_action_key

            # create a pd.DataFrame with relevant data for a single trial
            # n_actions_taken = trial_data['n actions taken'][trial_data.index[-1]]
            _trial_history = list()
            for idx in trial_data.index:

                # create a SARS tuple
                start_location = tuple(trial_data[u'Start Location'][idx])
                end_location = tuple(trial_data[u'End Location'][idx])
                s = (start_location, int(trial_data[u'context'][idx]))  # use zero-based indexing for python
                sp = (end_location, int(trial_data[u'context'][idx]))
                aa = abstract_action_key[trial_data[u'action'][idx]]
                r = int(trial_data['In goal'][idx])

                if str(int(trial_data['key-press'][idx])) in self.Task.keyboard_action_code.keys():
                    a = self.Task.keyboard_action_code[str(int(trial_data['key-press'][idx]))]
                    self.experience.append(tuple([s, a, aa, r, sp]))


                _trial_history.append(pd.DataFrame({
                    'Reward': int(trial_data[u'In goal'][idx]),
                    'Context': int(trial_data[u'context'][idx]),
                    'StartLocation': [tuple(trial_data[u'Start Location'][idx])],
                    'EndLocation': [tuple(trial_data[u'End Location'][idx])],
                    'StepNumber': trial_data['n actions taken'][idx],
                    'KeyPress': int(trial_data['key-press'][idx])
                }, index=[idx]))

            _trial_history = pd.concat(_trial_history)
            _trial_history.index = range(len(_trial_history))

            self.trial_history.append(_trial_history)


class GenerativeModelData(object):
    """ this is a data structure that holds all of the subjects trial by trial data, used for fitting etc.
    """

    def __init__(self, subject_data, grid_world_size=(3, 3), n_abstract_actions=4,
                 primitive_actions=(0, 1, 2, 3, 4, 5, 6, 7)):

        self.Task = make_task_from_generative(subject_data, grid_world_size, n_abstract_actions, primitive_actions)

        # need the equivalent of SARS tuples
        # inverse_state_loc_key = self.

        # need: start position, end position & keypress for each step
        self.trial_history = list()
        self.experience = list()
        for ii in range(self.Task.n_trials):

            trial_data = subject_data[subject_data.TrialNumber == ii]

            # needed for encoding cardinal directions numerically
            abstract_action_key = self.Task.trials[ii].abstract_action_key

            # create a pd.DataFrame with relevant data for a single trial
            # n_actions_taken = trial_data['n actions taken'][trial_data.index[-1]]
            _trial_history = list()
            for idx in trial_data.index:

                # create a SARS tuple
                start_location = tuple(trial_data[u'Start Location'][idx])
                end_location = tuple(trial_data[u'End Location'][idx])
                s = (start_location, int(trial_data[u'context'][idx]))  # use zero-based indexing for python
                sp = (end_location, int(trial_data[u'context'][idx]))
                aa = abstract_action_key[trial_data[u'action'][idx]]
                r = int(trial_data['In goal'][idx])

                if str(int(trial_data['key-press'][idx])) in self.Task.keyboard_action_code.keys():
                    a = self.Task.keyboard_action_code[str(int(trial_data['key-press'][idx]))]
                    self.experience.append(tuple([s, a, aa, r, sp]))


                _trial_history.append(pd.DataFrame({
                    'Reward': int(trial_data[u'In goal'][idx]),
                    'Context': int(trial_data[u'context'][idx]),
                    'StartLocation': [tuple(trial_data[u'Start Location'][idx])],
                    'EndLocation': [tuple(trial_data[u'End Location'][idx])],
                    'StepNumber': trial_data['n actions taken'][idx],
                    'KeyPress': int(trial_data['key-press'][idx])
                }, index=[idx]))

            _trial_history = pd.concat(_trial_history)
            _trial_history.index = range(len(_trial_history))

            self.trial_history.append(_trial_history)
