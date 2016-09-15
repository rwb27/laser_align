import nplab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import proj3d
import numpy as np


nplab.datafile.set_current(r"C:\Users\a-amb\OneDrive - University Of "
                           r"Cambridge\tests.hdf5", mode="r")
df = nplab.current_datafile()
ax = plt.axes(projection='3d')

for i in range(32):
    ser1 = df['AdaptiveHillWalk/AdaptiveHillWalk_54/brightness_{}'.format(i)][
           :, ...]
    length = ser1[:, 0].size
    colours = np.ones(length)
    ax.scatter(ser1[:, 1], ser1[:, 2], ser1[:, 3], s=ser1[:, 4] / 10**(
        ser1[:, 8]/10) * 300000, label='{}'.format(ser1[0, 0]))

plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.show()
plt.axis('equal')
