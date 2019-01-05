import gym
import gridworlds
import numpy as np
import matplotlib.pyplot as plt

"""
For the purpose of implementation, we are alloting 6 of 8 options for each state as we know that the other 2 options will be infeasible. EAch state can have atmost 6 feasible options.

Options are identified by 4 target states. Depending on the starting state, a target state will either correspond to one option or another making total number of options 8.
The first 4 elements of self.options[state] are the primitive actions. The last 2 elements of self.options[state] contains the indices of the target state in target_state array, that are feasible from the 'state'. 
"""

WALL = 6
GOAL = 5

COLORS = {0:[1.0,1.0,1.0], 1:[0.6,0.6,0.6], \
		2:[0.3,0.3,0.3], 3:[0.1,0.1,0.1], \
		4:[1.0,0.0,0.0], 5:[1.0,0.0,1.0], \
		7:[1.0,1.0,0.0]}

target_state = [(3, 6), (7, 9), (10, 6), (6, 2)]
action_idx = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}


class Qlearning:
	"""
	Q learning algorithm
	"""

	def __init__(self, env, gamma = 0.9, alpha = 0.1, intra_option = False, verbose = False):
		self.state_dim = env.observation_space.n
		self.action_dim = env.action_space.n
		self.options = np.zeros((self.state_dim, self.action_dim + 2))
		self.Q = np.zeros((self.state_dim, self.action_dim + 2))
		self.gamma = gamma
		self.alpha = alpha
		self.eps = 0.1 
		self.env = env
		self.env.reset()
		self.env.verbose = verbose
		self.set_options() 
		self.intra_option = intra_option


	def set_options(self):
		# Rooms are indexed only to identify the available options for the states and are not in anyway used for learning optimal policy
		# Room 1 will have target_state index 0 and 3
		# Room 2 will have target_state index 1 and 0
		# Room 3 will have target_state index 2 and 1
		# Room 4 will have target_state index 3 and 2
		for st in range(self.state_dim):
			i, j = int(st / self.env.cols), st % self.env.cols
			self.options[st] = np.arange(self.action_dim + 2)
			idx = self.env.start_grid_map[i, j]
			left_idx = self.env.start_grid_map[i, j - 1]
			up_idx = self.env.start_grid_map[i - 1, j]

			if (i, j) in target_state:
				idx = target_state.index((i, j)) + 1
				self.options[st, 4:] = np.array((idx % 4, (idx + 2) % 4))
			elif idx < 6:
				self.options[st, 4:] = np.array(((idx + 3) % 4, (idx + 2) % 4))

	def get_option(self, state):
		if np.random.binomial(1, self.eps) == 1:
			return np.random.randint(0, self.env.action_space.n + 2) # +2 for options
		mx = max(self.Q[state, :])
		idx = np.where(self.Q[state, :] == mx)[0]
		return np.random.choice(idx)

	def terminal_state(self, curr_state, start_room_idx, start_state):
		state = int(curr_state / self.env.cols), curr_state % self.env.cols
		curr_room_idx = self.env.start_grid_map[state]
		return (not curr_room_idx == start_room_idx) and (not state == start_state)


	def execute_option(self, curr_state, option_idx):
		if (option_idx < 4):  # option corresponds to a primitive action
			nxt_state, rwd, _, _ = self.env.step(option_idx)
			return nxt_state, rwd, 1.0, {}

		# Hard coded the optimal policy of options for all starting states

		start_state = (int(curr_state / self.env.cols), curr_state % self.env.cols)
		start_room_idx = self.env.start_grid_map[start_state]
		t = target_state[int(self.options[curr_state, option_idx])]
		target_idx = int(self.options[curr_state, option_idx])
		if (start_room_idx == 0):  # If the option is starting from a hallway
			idx1 = target_state.index(start_state)
			idx2 = int(self.options[curr_state, option_idx])
			if (idx1 == (idx2 + 1) % 4): 
				start_room_idx = idx1 + 1
			else:
				start_room_idx = idx2 + 1

		Ro = 0.0
		g = 1.0
		num_steps = 0.0
		while not self.terminal_state(curr_state, start_room_idx, start_state):
			self.env.done = False  # Incase the option's trajectory has passed through goal
			curr_a = None
			state = (int(curr_state / self.env.cols), curr_state % self.env.cols)
			if (state[0] != t[0] and self.env.start_grid_map[state[0] + np.sign(t[0] - state[0]), state[1]] != WALL): 
				# Get inside if and only if the new state when moved in vertical direction is not a wall
				action = (np.sign(t[0] - state[0]), 0)
			else:
				action = (0, np.sign(t[1] - state[1]))
			curr_a = action_idx[action]
			nxt_state, rwd, _, _ = self.env.step(curr_a)

			if (self.intra_option):
				# Need to update all possible options that share the same experience

				# Updating Q value of primitive action. 
				# For this case we have to set the target Q value as max among ALL options of Q[next_state, options] + rwd
				self.Q[curr_state, curr_a] = (1.0 - self.alpha) * self.Q[curr_state, curr_a] + self.alpha * (rwd + self.gamma * max(self.Q[nxt_state, :])) 

				# Finding the index in which the current option is present in the corresponding self.option array of current state
				curr_o = None
				if (self.options[curr_state, 4] == target_idx): curr_o = 4
				else: curr_o = 5
				# Finding the index in which the current option is present in the corresponding self.option array of next state
				nxt_o = None
				if (self.options[nxt_state, 4] == target_idx): nxt_o = 4
				else: nxt_o = 5

				# updating Q value of the currently chosen multi-step option 
				# (that is the only multi-step option compatible with current experience)
				if not self.terminal_state(nxt_state, start_room_idx, start_state):
					# If next state is not a terminal state, then update the Q value using the same option value of the next state
					self.Q[curr_state, curr_o] = (1.0 - self.alpha) * self.Q[curr_state, curr_o] + self.alpha * (rwd + self.gamma * self.Q[nxt_state, nxt_o]) 
				else:
					# If next state is terminal state, then update the Q value using the max Q value of the next state
					self.Q[curr_state, curr_o] = (1.0 - self.alpha) * self.Q[curr_state, curr_o] + self.alpha * (rwd + self.gamma * max(self.Q[nxt_state, :])) 

			Ro += rwd * g 
			g *= self.gamma
			curr_state = nxt_state
			num_steps += 1.0

		return curr_state, Ro, num_steps, {}


	def run(self, num_episodes):
		
		avg_steps = []
		avg_reward = []
		self.Q = np.zeros((self.state_dim, self.action_dim + 2))
		for i in range(0, int(num_episodes)):
			if (i % 50 == 0): print("Episode = {}".format(i))
			self.env.reset()
			curr_state = self.env.current_state()

			num_steps = 1.0

			while not self.env.done:
				curr_o = self.get_option(curr_state)
				nxt_state, r, k, _ = self.execute_option(curr_state, curr_o)

				if not self.intra_option:
					self.Q[curr_state, curr_o] = (1.0 - self.alpha) * self.Q[curr_state, curr_o] + self.alpha * (r + (self.gamma**k) * max(self.Q[nxt_state, :]))

				curr_state = nxt_state
				num_steps += k
			
			avg_steps.append(num_steps)

		return np.array(avg_steps), np.array(avg_reward), self.Q


class SARSA:

	def __init__(self, env, gamma = 0.9, alpha = 0.15, Lambda = 0, verbose = False):
		self.Q = np.zeros((env.observation_space.n, env.action_space.n)) 
		self.e = np.zeros((env.observation_space.n, env.action_space.n)) # eligibility trace
		self.gamma = gamma
		self.alpha = alpha
		self.Lambda = Lambda
		self.eps = 0.05
		self.env = env
		self.env.reset()
		self.env.verbose = verbose

	def get_action(self, state):
		if np.random.binomial(1, self.eps) == 1:
			return np.random.randint(0, self.env.action_space.n)
		mx = max(self.Q[state, :])
		idx = np.where(self.Q[state, :] == mx)[0]
		return np.random.choice(idx)

	def run(self, num_episodes):
		avg_steps = []
		avg_reward = []
		self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n)) 
		for i in range(num_episodes):
			print("Episode = {}".format(i))
			self.e = np.zeros((self.env.observation_space.n, self.env.action_space.n))
			self.env.reset()
			curr_state = self.env.current_state()
			curr_a = self.get_action(curr_state)
			num_steps = 1.0
			tot_reward = 0

			while (self.env.done == False):
				nxt_state, r, _, _ = self.env.step(curr_a)
				nxt_a = self.get_action(nxt_state)
				tot_reward += r

				if self.Lambda == 0:
					self.Q[curr_state, curr_a] = (1.0 - self.alpha) * self.Q[curr_state, curr_a] + self.alpha * (r + self.gamma * self.Q[nxt_state, nxt_a])
				else:
					TD_error = r + self.gamma * self.Q[nxt_state, nxt_a] - self.Q[curr_state, curr_a]
					self.e[curr_state, curr_a] += 1.0
					self.Q += self.alpha * TD_error * self.e
					self.e *= self.gamma * self.Lambda

				curr_state, curr_a = nxt_state, nxt_a
				num_steps += 1.0
			
			avg_reward.append(tot_reward)
			avg_steps.append(num_steps)

		return np.array(avg_steps), np.array(avg_reward)

	def show_policy(self):
		# Should be called after self.run method
		policy_image = np.ones(self.env.obs_shape)
		gs0 = int(policy_image.shape[0] / self.env.grid_map_shape[0])
		gs1 = int(policy_image.shape[1] / self.env.grid_map_shape[1])
		for i in range(self.env.grid_map_shape[0]):
			for j in range(self.env.grid_map_shape[1]):
				for k in range(3):
					idx = self.env.start_grid_map[i, j]
					policy_image[i * gs0 : (i + 1) * gs0, j * gs1 : (j + 1) * gs1, k] = COLORS[idx][k]
					policy_image[i * gs0, j * gs1 : (j + 1) * gs1, k] = 0
					policy_image[(i + 1) * gs0, j * gs1 : (j + 1) * gs1, k] = 0
					policy_image[i * gs0 : (i + 1) * gs0, j * gs1, k] = 0
					policy_image[i * gs0 : (i + 1) * gs0, (j + 1) * gs1, k] = 0

		fig = plt.figure(3)
		plt.clf()
		plt.imshow(policy_image)

		rows = self.env.grid_map_shape[0]
		cols = self.env.grid_map_shape[1]
		for i in range(rows):
			for j in range(cols):
				action = np.argmax(self.Q[i * cols + j, :])
				if action == 0:
					plt.arrow((j + 1) * gs1 - gs1 / 2, (i + 1) * gs0 - gs0 / 4, 0, -gs0 / 2, head_width = 1.5, color = 'b', length_includes_head = True)
				elif action == 2:
					plt.arrow((j + 1) * gs1 - gs1 / 2, (i + 1) * gs0 - int(3.0 * gs0 / 4.0), 0, gs0 / 2, head_width = 1.5, color = 'b', length_includes_head = True)
				elif action == 1:
					plt.arrow(j * gs1 + gs1 / 4, (i + 1) * gs0 - gs0 / 2, gs1 / 2, 0, head_width = 1.5, color = 'b', length_includes_head = True)
				else:
					plt.arrow(j * gs1 + int(3.0 * gs1 / 4.0), (i + 1) * gs0 - gs0 / 2, -gs1 / 2, 0, head_width = 1.5, color = 'b', length_includes_head = True)

		plt.show()


