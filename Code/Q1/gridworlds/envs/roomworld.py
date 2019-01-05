import os
import sys
import gym
import time
import copy
import numpy as np
from gym import spaces
from PIL import Image as Image
import matplotlib.pyplot as plt

COLORS = {0:[1.0,1.0,1.0], 1:[1.0,1.0,1.0], \
		2:[1.0,1.0,1.0], 3:[1.0,1.0,1.0], \
		4:[1.0,1.0,1.0], 5:[1.0,0.0,0.0], \
		6:[0.1,0.1,0.1], 7:[0.0,0.0,1.0]}
# Color 7 corresponds to the current state of the agent. 
# Color 6 corresponds to wall
# Color 5 corresponds to the terminal state
# All other colors correspond to free space

class RoomWorldEnv(gym.Env):

	def __init__(self):
		self.rows = 13
		self.cols = 13
		self.action_space = spaces.Discrete(4)  # 4 primitive actions, 4 terminal states for 8 options
		self.observation_space = spaces.Discrete(self.rows * self.cols)
		self.reward_range = (0, 2)

		self.obs_shape = [131, 131, 3]
		self.state_reward = [0, 0, 0, 0, 0, 1, 0, 0]
		self.moves = {     # actions 0 to 3 correspond to hallway exit options
				0: (-1, 0),  # north
				1: (0, 1),   # west
				2: (1, 0),   # south
				3: (0, -1),  # east
				}

		''' initialize system state ''' 
		this_file_path = os.path.dirname(os.path.realpath(__file__))
		self.grid_map_path = os.path.join(this_file_path, 'plan1.txt')        
		self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map
		self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
		self.observation = self._gridmap_to_observation(self.start_grid_map)
		self.grid_map_shape = self.start_grid_map.shape

		self.terminal_state = self._get_agent_target_state(self.start_grid_map)
		self.terminal_reward = 1
		self.step_reward = 0

		self.verbose = False
		self.done = False
		self.reset()


	def _read_grid_map(self, grid_map_path):
		grid_map = open(grid_map_path, 'r').readlines()
		grid_map_array = []
		for k1 in grid_map:
			k1s = k1.split(' ')
			tmp_arr = []
			for k2 in k1s:
				try:
					tmp_arr.append(int(k2))
				except:
					pass
			grid_map_array.append(tmp_arr)
		grid_map_array = np.array(grid_map_array)
		return grid_map_array


	def _gridmap_to_observation(self, grid_map, obs_shape=None, state_list = None):
		# Converts gridmap array to image array
		if obs_shape is None:
			obs_shape = self.obs_shape
		gs0 = int(obs_shape[0] / grid_map.shape[0])
		gs1 = int(obs_shape[1] / grid_map.shape[1])

		if state_list != None:
			# Make changes only to the given list of states
			for states in state_list:
				i, j = states
				for k in range(3):
					this_value = COLORS[grid_map[i, j]][k]
					self.observation[i * gs0 : (i + 1) * gs0, j * gs1 : (j + 1) * gs1, k] = this_value
			return

		observation = np.random.randn(*obs_shape) * 0.0
		for i in range(grid_map.shape[0]):
			for j in range(grid_map.shape[1]):
				for k in range(3):
					this_value = COLORS[grid_map[i, j]][k]
					observation[i * gs0 : (i + 1) * gs0, j * gs1 : (j + 1) * gs1, k] = this_value

		return observation


	def _get_agent_target_state(self, start_grid_map):
		#target state is the position where number '4' is present in the plan.txt file
		target_state = []
		for i in range(start_grid_map.shape[0]):
			for j in range(start_grid_map.shape[1]):
				this_value = start_grid_map[i, j]
				if this_value == 5:
					target_state.append((i, j))
		
		if (target_state == None):
			sys.exit('Start or target state not specified')
		print (target_state)
		return target_state[0]


	def step(self, action):
		x, y = self.moves[action]
		if (np.random.binomial(1, 4.0 / 9.0) == 0):  
			# Random action with probability of picking an action being 0.1 / 3. 
			# Hence the probability of picking the right action becomes 0.9 and probability of picking each of the incorrect action is 0.1 / 3 
			x, y = self.moves[np.random.randint(0, 4)]

		nxt_state = min(self.rows - 1, max(self.state[0] + x, 0)), min(self.cols - 1, max(self.state[1] + y, 0))
		if (self.start_grid_map[nxt_state] == 6):  # If wall
			nxt_state = self.state

		if (self.verbose and nxt_state != self.state):  # Update image only if the state has changed
			self.current_grid_map[self.state[0], self.state[1]] = self.start_grid_map[self.state[0], self.state[1]]
			self.current_grid_map[nxt_state[0], nxt_state[1]] = 7
			self._gridmap_to_observation(self.current_grid_map, state_list = [self.state, nxt_state])
		self.render()

		self.state = nxt_state
		self.done = (self.state == self.terminal_state)
		reward = self.state_reward[self.start_grid_map[self.state[0], self.state[1]]]
	
		return self.current_state(), reward, self.done, None

	def reset(self):
		self.state = (1, 1)
		self.current_grid_map = copy.deepcopy(self.start_grid_map)
		self.current_grid_map[self.state[0], self.state[1]] = 7
		self.observation = self._gridmap_to_observation(self.start_grid_map)
		self.done = False
		self.render()
		return self.state
		
	def render(self):
		if self.verbose == False:
			return
		img = self.observation
		fig = plt.figure(1)
		plt.clf()
		plt.imshow(img)
		fig.canvas.draw()
		plt.pause(0.0000001)
		return

	def current_state(self):
		return self.state[0] * self.cols + self.state[1]
