import gym
import gridworlds
import numpy as np
from algorithms import Qlearning
import matplotlib.pyplot as plt

env = gym.make('roomworld-v0')

learner = Qlearning(env, verbose = False)

num_episodes = 1000
avg_steps, avg_rwd = learner.run(num_episodes)

plt.figure(1)
plt.plot(np.arange(num_episodes), avg_steps)
plt.xlabel("Episodes")
plt.ylabel("Average steps")
plt.show()

plt.figure(2)
plt.plot(np.arange(num_episodes), avg_steps)
plt.xlabel("Episodes")
plt.ylabel("Average reward")
plt.show()
