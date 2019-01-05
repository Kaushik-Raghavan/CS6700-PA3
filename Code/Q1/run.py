from algorithms import Qlearning
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import gridworlds
import numpy as np
import joblib
import gym

env = gym.make('roomworld-v0')


def get_derivables(learner, num_episodes, num_expts, show_policy = False, show_plots = True):
	''' num_episodes is number of episodes ''' 
	''' num_expts is number of experiments to average the performance over '''
	''' if show_policy is True, an image will be displayed showing optimal policy found by the agent after all episodes '''
	x_axis = np.arange(num_episodes)
	avg_steps = np.zeros(num_episodes)
	avg_reward = np.zeros(num_episodes)
	Q = None
	for i in range(num_expts):
		print (i)
		steps, rwd, Q = learner.run(num_episodes)
		avg_steps += steps

	avg_steps /= float(num_expts)

	if (show_plots):
		plt.figure(1)
		plt.loglog(x_axis, avg_steps)
		plt.ylabel("Average steps per episode")
		plt.xlabel("episodes")

		fig = plt.figure()
		ax = fig.gca(projection = '3d')

		rows = learner.env.rows
		cols = learner.env.cols
		Qvalue = np.zeros((rows, cols))
		for i in range(rows):
			for j in range(cols):
				Qvalue[i, j] = np.max(Q[i * cols + j])

		X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
		ax.plot_surface(X, Y, np.transpose(Qvalue))

		plt.show()
		np.savetxt("QvaluesIntra1.txt", Qvalue, fmt = '%0.3f', delimiter = ' ')
		joblib.dump(avg_steps1, "avg_stepsIntra1")

	return avg_steps


learner = Qlearning(env, intra_option = True)
get_derivables(learner, 5000, 10, False)
#get_derivables(SARSA(env, Lambda = 0.1, verbose = False), 100, 1, False)

"""
avg_steps = []
avg_reward = []
lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for l in lambdas:
	print ("    Lambda = {}".format(l))
	steps, reward = get_derivables(SARSA(env, Lambda = l, verbose = False), 25, 10, show_plots = False)
	avg_steps.append(steps[24])
	avg_reward.append(reward[24])

plt.figure(3)
plt.plot(lambdas, avg_steps)
plt.title(r"Average steps after 25 episodes for different $\lambda$")
plt.ylabel("Average steps to reach the goal (averaged over 10 experiments)")
plt.xlabel(r"$\lambda$")

plt.figure(2)
plt.plot(lambdas, avg_reward)
plt.title(r"Average reward after 25 episodes for different $\lambda$")
plt.ylabel("Average reward per episode(averaged over 10 experiments)")
plt.xlabel(r"$\lambda$")

plt.show()
"""
