import copy

import numpy as np
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
                 state_location_key=None, n_abstract_actions=4, n_primitive_actions=8):
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
        self.state_successor = np.ndarray((n_states, n_abstract_actions), dtype=int)
        for s in range(n_states):

            x, y = self.inverse_state_loc_key[s]

            # cycle through movement, check for both walls and for
            for movement, (dx, dy) in self.cardinal_direction_key.iteritems():
                aa = self.abstract_action_key[movement]

                # check if the movement stays on the grid
                if (x + dx, y + dy) not in self.state_location_key.keys():
                    self.transition_function[s, aa, s] = 1
                    self.state_successor[s, aa] = s

                elif (x, y) in wall_list:
                    # check if the movement if blocked by a wall
                    if wall_key[(x, y)] == movement:
                        self.transition_function[s, aa, s] = 1
                        self.state_successor[s, aa] = s
                    else:
                        sp = self.state_location_key[(x + dx, y + dy)]
                        self.transition_function[s, aa, sp] = 1
                        self.state_successor[s, aa] = sp
                else:
                    sp = self.state_location_key[(x + dx, y + dy)]
                    self.transition_function[s, aa, sp] = 1
                    self.state_successor[s, aa] = sp

        # make reward function
        self.reward_function = np.zeros((n_states, n_states), dtype=float)
        self.reward_function[:, :] = 0
        self.reward_function[:, self.state_location_key[goal]] = 1.0

        # store the action map
        self.action_map = {int(key): value for key, value in action_map.iteritems()}
        self.n_primitive_actions = n_primitive_actions

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

        # define the transition function in terms of primitive actions
        self.primitive_transition_function = np.zeros((n_states, self.n_primitive_actions, n_states))
        for a in range(self.n_primitive_actions):
            if a not in self.keys_used:
                for s in range(n_states):
                    self.primitive_transition_function[s, a, s] = 1.0

        for (loc, a), locp in self.successor_function.iteritems():
            s = self.state_location_key[loc]
            sp = self.state_location_key[locp]
            self.primitive_transition_function[s, a, sp] = 1.0

    def move(self, key_press):
        """
        :param key_press: int key-press
        :return: tuple (((x, y), c), a, aa, r, ((xp, yp), c))
        """
        prev_location = self.current_location
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

        return (prev_location, self.context), key_press, aa, r, (new_location, self.context)

    def goal_check(self):
        return self.current_location == self.goal_location

    def get_state(self):
        return self.current_location, self.context

    def draw_state(self, fig=None, ax=None, draw_goal=True):
        x, y = self.current_location

        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))

        self._draw_grid(ax=ax)
        self._draw_wall(ax=ax)

        ax.plot([x, x], [y, y], 'bo', markersize=15)

        # draw the goal location:
        if draw_goal:
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
                 primitive_actions=(0, 1, 2, 3, 4, 5, 6, 7),
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
            return self.current_trial.get_state()

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

    def get_mapping_function(self, aa):
        mapping = np.zeros((self.n_primitive_actions, self.n_abstract_actions), dtype=float)
        for a, dir_ in self.current_trial.action_map.iteritems():
            aa0 = self.current_trial.abstract_action_key[dir_]
            mapping[a, aa0] = 1

        return np.squeeze(mapping[:, aa])

    def get_transition_function(self):
        return self.current_trial.primitive_transition_function

    def get_reward_function(self):
        rewards = np.zeros(self.n_states, dtype=float)

        # average over all entering states...
        for sp in range(self.n_states):
            rewards[sp] = self.current_trial.reward_function[:, sp].mean() > 0
        return rewards


class Room(object):
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
        self.reward_function[:, :] = 0
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

    def draw_state(self, fig=None, ax=None, draw_goal=True):
        x, y = self.current_location

        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))

        self._draw_grid(ax=ax)
        self._draw_wall(ax=ax)

        ax.plot([x, x], [y, y], 'bo', markersize=15)

        # draw the goal location:
        if draw_goal:
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


class RoomsProblem(object):
    """ This is a data structure that holds all of the trials a subject encountered in a format readable by the models.
    This is used primarily for the purposes of initialization of the agents.
    """

    def __init__(self, list_start_location, list_end_location, list_context, list_action_map,
                 grid_world_size=(3, 3),
                 n_abstract_actions=4,
                 primitive_actions=(0, 1, 2, 3, 4, 5, 6, 7),
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
            return self.current_trial.get_state()

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
        if (np.random.rand() <= hazard_rates[n_repeats]) | (context_presentations[current_context] < 1):

            # construct a list of available contexts to select
            _available_ctx = range(n_ctx)
            available_contexts = []
            for ctx in _available_ctx:
                if (context_presentations[ctx] > 0) & (ctx != current_context):
                    available_contexts += [ctx] * context_presentations[ctx]

            # randomly select one available context
            if available_contexts:  # check if empty!
                current_context = available_contexts[np.random.randint(len(available_contexts))]
                n_repeats = -1

        # update counters
        n_repeats += 1
        context_presentations[current_context] -= 1

    return context_order


def make_task(context_balance, context_goals, context_maps, hazard_rates, start_locations, grid_world_size,
              list_walls=None):

    # start locations are completely random
    list_start_locations = list()
    for ctx, n_reps in enumerate(context_balance):
        list_start_locations += [start_locations[np.random.randint(len(start_locations))] for _ in range(n_reps)]

    # use randomization function to shuffle contexts
    list_context = randomize_order(context_balance, hazard_rates)

    # list of goals and list of mappings depend on the context order
    list_goals = [context_goals[ctx] for ctx in list_context]
    list_maps = [context_maps[ctx] for ctx in list_context]

    # print list_context, list_goals, list_maps

    if list_walls is None:
        list_walls = [[]] * len(list_context)

    args = [list_start_locations, list_goals, list_context, list_maps]
    kwargs = dict(list_walls=list_walls, grid_world_size=grid_world_size)
    return Task(*args, **kwargs)