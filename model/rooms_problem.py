import numpy as np
import matplotlib.pyplot as plt
import time

"""
This is a variant of the grid worlds where goals are labeled and known to the agent, but their reward value
is not. This allows dissociating a movements towards a goal from raw stimulus-response associations and
can be used to force planning (both by moving the goals from trial to trial).

Reward associations are meant to be constant for each goal, but the location is meant to change from trial
to trial (here, GridWorld instance to GridWorld instance within a task)


"""


# code to make the grid world starts here!


class GridWorld(object):
    def __init__(self, grid_world_size, walls, action_map, goal_dict, start_location, context,
                 state_location_key=None, n_abstract_actions=4):
        """

        :param grid_world_size: 2x2 tuple
        :param walls: list of [x, y, 'direction_of_wall'] lists
        :param action_map: dictionary of from {a: 'cardinal direction'}
        :param goal_dict: dictionary {(x, y): ('label', r)}
        :param start_location: tuple (x, y)
        :param n_abstract_actions: int
        :return:
        """
        self.start_location = start_location
        self.current_location = start_location
        self.grid_world_size = grid_world_size
        self.context = int(context)
        self.walls = walls

        # need to create a transition function and reward function, which pretty much define the grid world
        n_states = grid_world_size[0] * grid_world_size[1]  # assume rectangle
        if state_location_key is None:
            self.state_location_key = \
                {(x, y): (y + x * grid_world_size[1]) for y in range(grid_world_size[1]) for x in
                    range(grid_world_size[0])}
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

        # set up goals
        self.goal_dictionary = goal_dict
        self.goal_locations = {loc: label for loc, (label, _) in goal_dict.iteritems()}
        self.goal_values = {label: r for _, (label, r) in goal_dict.iteritems()}

        # make the goal states self absorbing!!
        for loc in self.goal_locations.iterkeys():
            s = self.state_location_key[loc]
            self.transition_function[s, :, :] = 0.0
            self.transition_function[s, :, s] = 1.0

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

    def reset(self):
        self.current_location = self.start_location

    def move(self, key_press):
        """
        :param key_press: int key-press
        :return:
        """
        if key_press in self.keys_used:
            new_location = self.successor_function[self.current_location, key_press]
            # get the abstract action number
            aa = self.action_map[key_press]
        else:
            new_location = self.current_location
            aa = u'wait'

        # update the current location before returning
        self.current_location = new_location

        # goal check, and return goal id + reward
        if self.goal_check():
            goal_id, r = self.goal_probe()
            return aa, new_location, goal_id, r

        return aa, new_location, None, None

    def goal_check(self):
        if self.current_location in self.goal_dictionary.keys():
            return True
        return False

    def goal_probe(self):
        return self.goal_dictionary[self.current_location]

    def get_location(self):
        return self.current_location

    def get_goal_locations(self):
        return self.goal_locations

    def draw_state(self, fig=None, ax=None):
        x, y = self.current_location

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        self._draw_grid(ax=ax)
        self._draw_wall(ax=ax)

        ax.plot([x, x], [y, y], 'bo', markersize=12)

        # draw the goal locations:
        for loc, (label, _) in self.goal_dictionary.iteritems():
            ax.annotate(label, xy=(loc[0] - 0.25, loc[1] - 0.25), xytext=(loc[0] - 0.25, loc[1] - 0.25), size=14)
        return fig, ax

    def draw_move(self, key_press, fig=None, ax=None):
        fig, ax = self.draw_state(fig=fig, ax=ax)
        print "Current State:", self.get_location()

        # use the successor function!
        if key_press in self.keys_used:
            (xp, yp) = self.successor_function[self.current_location, key_press]
            print "Key Press:", str(key_press), "; Corresponding Movement:", self.action_map[key_press]
        else:
            (xp, yp) = self.current_location
            print "Key Press:", str(key_press), "; Corresponding Movement: wait"

        ax.plot([xp, xp], [yp, yp], 'go', markersize=14)

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
        # sns.set_style('white')
        for x in range(self.grid_world_size[0] + 1):
            for y in range(self.grid_world_size[1] + 1):
                ax.plot([-0.5, self.grid_world_size[0] - 0.5], [y - 0.5, y - 0.5], color=[0.85, 0.85, 0.85])
                ax.plot([x - 0.5, x - 0.5], [-0.5, self.grid_world_size[1] - 0.5], color=[0.85, 0.85, 0.85])

        # finish the plot!
        # sns.despine(left=True, bottom=True)
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
        # sns.set_style('white')

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

    # def get_true_reward_function(self, pi):

    def draw_policy(self, pi):
        # first use the gw_size to draw the most basic grid
        # sns.set_style('white')

        fig, ax = plt.subplots(figsize=(4, 4))
        self.draw_state(ax=ax)

        displacements = {u'up': (0.0, 0.75), u'left': (-0.75, 0.0), u'right': (0.75, 0.0), u'down': (0.0, -0.75)}

        for s in range(self.transition_function.shape[0]):
            aa = self.inverse_abstract_action_key[pi[s]]
            x, y = self.inverse_state_loc_key[s]
            dx, dy = displacements[aa]

            ax.arrow(x, y, dx, dy, width=0.01, color='k')

        self._draw_wall(ax=ax)

        # finish the plot!
        plt.show()


class Task(object):
    pass


class Experiment(Task):
    """ This is a data structure that holds all of the trials a subject encountered in a format readable by the models.
    This is used primarily for the purposes of initialization of the agents.
    """

    def __init__(self, list_start_location, list_goals, list_context, list_action_map,
                 grid_world_size=(6, 6),
                 n_abstract_actions=4,
                 primitive_actions=(72, 74, 75, 76, 65, 83, 68, 70),
                 list_walls=None,
                 ):
        """
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

        # count the number (and get the name) of the goals
        self.goals = set([g for g, _ in list_goals[0].itervalues()])
        self.goal_index = {g: ii for ii, g in enumerate(self.goals)}
        self.reverse_goal_index = {v: k for k, v in self.goal_index.iteritems()}
        self.n_goals = len(self.goals)

        # create a state location key
        self.state_location_key = \
            {(x, y): (y + x * grid_world_size[1]) for y in range(grid_world_size[1]) for x in range(grid_world_size[0])}

        self.inverse_state_loc_key = {value: key for key, value in self.state_location_key.iteritems()}

        # create a key-code between keyboard presses and action numbers
        self.keyboard_action_code = {unicode(keypress): a for a, keypress in enumerate(primitive_actions)}

        # for each trial, I need the walls, action_map and goal location and start location to define the grid world
        self.trials = list()
        if list_walls is None:
            list_walls = [list()] * self.n_trials

        for ii in range(self.n_trials):
            self.trials.append(
                GridWorld(
                    grid_world_size,
                    list_walls[ii],
                    list_action_map[ii],
                    list_goals[ii],
                    list_start_location[ii],
                    list_context[ii],
                    state_location_key=self.state_location_key,
                    n_abstract_actions=n_abstract_actions
                )
            )

        # set the current trial
        self.current_trial_number = 0
        self.current_trial = self.trials[0]
        assert type(self.current_trial) is GridWorld

        self.abstract_action_key = self.current_trial.abstract_action_key

        # store all of the action maps

        set_keys = set()
        for m in list_action_map:
            key = tuple((a, dir_) for a, dir_ in m.iteritems())

            set_keys.add(key)

        self.list_action_maps = []
        for s in set_keys:
            self.list_action_maps.append(dict())
            for key, dir_ in s:
                self.list_action_maps[-1][key] = dir_

    def goal_check(self):
        return self.current_trial.goal_check()

    def get_location(self):
        if self.current_trial is not None:
            return self.current_trial.get_location()

    def get_current_context(self):
        return self.current_trial.context

    def get_trial_number(self):
        return self.current_trial_number

    def get_transition_function(self):
        return self.current_trial.transition_function

    def get_current_gridworld(self):
        return self.current_trial

    def get_goal_locations(self):
        if self.current_trial is not None:
            return self.current_trial.get_goal_locations()

    def get_walls(self):
        if self.current_trial is not None:
            return self.current_trial.walls

    def get_action_map(self):
        if self.current_trial is not None:
            return self.current_trial.action_map

    def get_goal_index(self, goal):
        return self.goal_index[goal]

    def get_goal_values(self):
        goal_values = np.zeros(self.n_goals)
        for g, idx in self.goal_index.iteritems():
            goal_values[idx] = self.current_trial.goal_values[g]

        return goal_values

    def get_mapping_function(self, aa):
        mapping = np.zeros((self.n_primitive_actions, self.n_abstract_actions), dtype=float)
        for a, dir_ in self.current_trial.action_map.iteritems():
            aa0 = self.current_trial.abstract_action_key[dir_]
            mapping[a, aa0] = 1

        return np.squeeze(mapping[:, aa])

    def move(self, action):
        # self.current_trial.draw_move(key_press)
        aa, new_location, goal_id, r = self.current_trial.move(action)

        if goal_id is not None:
            self.start_next_trial()

        return aa, new_location, goal_id, r

    def move_with_trial_reset(self, action):
        # this is the move function to use for task where the trial resets after choosing the wrong goal
        aa, new_location, goal_id, r = self.current_trial.move(action)

        if goal_id is not None:
            if r > 0:
                self.start_next_trial()
            else:
                self.reset_trial()

        return aa, new_location, goal_id, r

    def end_check(self):
        if self.current_trial is None:
            return True
        return False

    def start_next_trial(self):
        self.current_trial_number += 1
        if self.current_trial_number < len(self.trials):
            self.current_trial = self.trials[self.current_trial_number]
        else:
            self.current_trial = None

    def reset_trial(self):
        self.current_trial.current_loaction = self.current_trial.start_location


class RoomsProblem(Task):

    def __init__(self, room_mappings, successor_function, reward_function,
                 list_start_locations,
                 list_door_locations,
                 grid_world_size=(6, 6),
                 n_abstract_actions=4,
                 primitive_actions=(72, 74, 75, 76, 65, 83, 68, 70),
                 list_walls=None
                 ):

        # create a state location key
        self.state_location_key = {
            (x, y): (y + x * grid_world_size[1])
            for y in range(grid_world_size[1])
            for x in range(grid_world_size[0])
            }

        self.rooms = dict()
        for r in range(len(room_mappings)):
            goal_dict = {l: (g, reward_function[r][g]) for g, l in list_door_locations[r].iteritems()}

            if list_walls is not None:
                walls = list_walls[r]
            else:
                walls = []

            self.rooms[r] = GridWorld(grid_world_size, walls, room_mappings[r], goal_dict,
                                      list_start_locations[r], r, state_location_key=self.state_location_key)

        self.current_room_number = 0
        self.current_room = self.rooms[0]
        self.n_rooms = len(self.rooms)
        self.rooms[None] = None  # augment rooms with end state
        self.successor_function = successor_function
        self.reward_function = reward_function

        self.trial_number = 1  # number of rooms the agent has visited
        self.goal_index = {
            g: ii for ii, g in
            enumerate(set([g for d in reward_function.itervalues() for g in d.keys()]))
                          }
        self.n_goals = len(self.goal_index.keys())
        self.n_abstract_actions = n_abstract_actions
        self.n_primitive_actions = len(primitive_actions)
        self.primitive_actions = primitive_actions
        self.abstract_action_key = self.current_room.abstract_action_key

    def move(self, action):

        aa, new_location, goal_id, r = self.current_room.move(action)
        if goal_id is not None:
            next_room = self.successor_function[self.current_room_number][goal_id]
            self.current_room = self.rooms[next_room]
            self.current_room_number = next_room
            self.reset_room()
            self.trial_number += 1

        return aa, new_location, goal_id, r

    def reset_room(self):
        if self.current_room is not None:
            self.current_room.reset()

    def end_check(self):
        if self.current_room_number is None:
            return True
        return False

    def get_current_room(self):
        return self.current_room_number

    def get_current_context(self):
        return self.get_current_room()

    def get_transition_function(self):
        return self.current_room.transition_function

    def get_location(self):
        return self.current_room.get_location()

    def get_trial_number(self):
        return self.trial_number

    def get_current_gridworld(self):
        return self.current_room

    def get_goal_locations(self):
        if self.current_room is not None:
            return self.current_room

    def get_walls(self):
        if self.current_room is not None:
            return self.current_room.walls

    def get_action_map(self):
        if self.current_room is not None:
            return self.current_room.action_map

    def get_goal_values(self):
        goal_values = np.zeros(self.n_goals)
        for g, idx in self.goal_index.iteritems():
            goal_values[idx] = self.current_room.goal_values[g]
        return goal_values

    def get_goal_index(self, goal):
        return self.goal_index[goal]

    def get_mapping_function(self, aa):
        mapping = np.zeros((self.n_primitive_actions,
                            self.n_abstract_actions), dtype=float)
        for a, dir_ in self.current_room.action_map.iteritems():
            aa0 = self.current_room.abstract_action_key[dir_]
            mapping[a, aa0] = 1

        return np.squeeze(mapping[:, aa])
