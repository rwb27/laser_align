import nplab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import proj3d
import numpy as np


nplab.datafile.set_current(r"C:\Users\a-amb\OneDrive - University Of "
                           r"Cambridge\tests.hdf5", mode="r")
df = nplab.current_datafile()

ser1 = df['AdaptiveHillWalk/AdaptiveHillWalk_40/brightness_1'][:, ...]
#ser2 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_0'][:, ...]
#ser3 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_1'][:, ...]
#ser4 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_2'][:, ...]
#ser5 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_3'][:, ...]

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.scatter(ser1[12:19, 2], ser1[12:19, 4])

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.scatter(ser1[12:19, 2], ser1[12:19, 0])
plt.show()

print ser1[:, ...]

#mpl.rcParams['legend.fontsize'] = 10

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#
##for ser in [ser1, ser2, ser3, ser4, ser5]:
##    x = ser[:, 1]
##    y = ser[:, 2]
##    z = ser[:, 3]
##    ax.plot(x, y, z)
##    for label, x, y, z in zip(ser[:, 4], x, y, z):
##        ax.annotate(str(label), xy=(x, y, z), textcoords='data')
##
##plt.show()
#
#
#
## if this code is placed inside a function, then
## we must use a predefined global variable so that
## the update function has access to it. I'm not
## sure why update_positions() doesn't get access
## to its enclosing scope in this case.
#global labels_and_points
#labels_and_points = []
#
#ser = ser1
#labels = np.array([(ser[i, 0], ser[i, 4]) for i in xrange(ser[:, 0].size)])
#xs = ser[:, 1]
#ys = ser[:, 2]
#zs = ser[:, 3]
#sc = ax.scatter(xs,ys,zs)
#
#for txt, x, y, z in zip(labels, xs, ys, zs):
#    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
#    label = plt.annotate(txt, xy = (x2, y2), textcoords='data')
#    labels_and_points.append((label, x, y, z))
#
#
#def update_position(e):
#    for label, x, y, z in labels_and_points:
#        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
#        label.xy = x2,y2
#        label.update_positions(fig.canvas.renderer)
#    fig.canvas.draw()
#
#fig.canvas.mpl_connect('motion_notify_event', update_position)
#
#plt.show()