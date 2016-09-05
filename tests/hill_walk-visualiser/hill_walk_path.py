import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nplab

nplab.datafile.set_current(r"C:\Users\a-amb\OneDrive - University Of "
                           r"Cambridge\sophistic_hill_walk.h5", mode="r")
df = nplab.current_datafile()
data = df['HillWalk/HillWalk_4/hill_walk_brightness'][:, ...]

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = data[:, 0], data[:, 1]
Z = data[:, 2]
ax.plot(X, Y, Z)

plt.show()
