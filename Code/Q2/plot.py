import matplotlib.pyplot as plt
import numpy as np
import csv

avg_steps_file = open("AverageRewards60-60.csv", "r").readlines()
avg_steps = []
for k in avg_steps_file:
	avg_steps.append(float(k))

plt.figure(1)
plt.title("Average reward over 100 episodes")
plt.xlabel("Episodes")
plt.ylabel("Avg number of steps/rewards")
plt.plot(np.arange(len(avg_steps)), avg_steps)
plt.show()
