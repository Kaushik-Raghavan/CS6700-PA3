from algorithms import Qlearning
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import gridworlds
import numpy as np
import joblib
import gym


values = open("Qvalues.txt", 'r').readlines()
Qvalues = []
for q in values:
	qs = q.split(' ')
	tmp_arr = []
	for q2 in qs:
		tmp_arr.append(float(q2))
	Qvalues.append(tmp_arr)

Qvalues = np.transpose(np.array(Qvalues)) 
fig = plt.figure()
ax = fig.gca(projection = '3d')
X, Y = np.meshgrid(np.arange(13), np.arange(13))
ax.plot_surface(X, Y, Qvalues)
plt.show()

