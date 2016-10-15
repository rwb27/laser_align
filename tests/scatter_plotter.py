import h5py as h
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

f = h.File('C:/Users/Abhishek/OneDrive - University Of '
           'Cambridge/data/tests.hdf5',
           "r")
#experiment_group = f['HillWalk']
# process by multiplying by gain appropriately.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#for group in experiment_group.values():
group = f['/AdaptiveHillWalk/AdaptiveHillWalk_41']
whole_run = np.zeros((1, 9))
if len(group.values()) == 0:
    pass
for dset in group.values():#[int(len(group.values())/2.):5*int(len(
    # #group.values())/6.)]
    #if dset.size != 0 and dset.name[:51] == \
    #        '/AdaptiveHillWalk/AdaptiveHillWalk_0/hill_walk_brightness':
        whole_run = np.vstack((whole_run, dset))
        print dset.name
whole_run = whole_run[1:, ...]
print whole_run
ax.scatter(whole_run[:, 1], whole_run[:, 2], whole_run[:, 3], s=whole_run[:,
                                                                4]/ 10 ** (
    whole_run[:, 8] / 10) * 30000,
c=whole_run[0:, 1], cmap='viridis', lw=0, zorder=1)

plt.tick_params(direction='out', labelsize=12)
plt.grid(True)
ax.ticklabel_format(useOffset=False)
ax.set_xlabel('X', fontsize=14, fontweight='bold')
ax.set_ylabel('Y', fontsize=14, fontweight='bold')
ax.set_zlabel('Z', fontsize=14, fontweight='bold')
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10
plt.show()



