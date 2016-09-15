import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import nplab

def scan_direction(dset):
    """Return the dimension along which we were scanning."""
    for i in range(3):
        if np.any(dset[:, i+1] != dset[0, i+1]):
            return i
    raise ValueError("Can't find the scan direction: no axis is moving!")


def plot_all_alignments_in_direction(direction, dsets):
    f, ax = plt.subplots(1, 1)
    for d in dsets:
        if scan_direction(d) == direction:
            ax.plot(d[:, 1 + direction], d[:, 4] * 10 ** (7 - d[:, 8] / 10),
                    'o-')

def plot_3d(dsets):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for i in range(32):
        ser1 = df['AdaptiveHillWalk/AdaptiveHillWalk_54/brightness_{}'.format(
            i)][
               :, ...]
        length = ser1[:, 0].size
        colours = np.ones(length)
        ax.scatter(ser1[:, 1], ser1[:, 2], ser1[:, 3], s=ser1[:, 4] / 10 ** (
            ser1[:, 8] / 10) * 300000, label='{}'.format(ser1[0, 0]))

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.legend()

if __name__ == '__main__':
    try:
        nplab.datafile.set_current(r"C:\Users\a-amb\OneDrive - University Of "
                                   r"Cambridge\tests.hdf5")
        df = nplab.current_datafile()
        dsets = df['AdaptiveHillWalk/AdaptiveHillWalk_58'].numbered_items(
            'brightness')

        plot_3d(dsets)

        for i in range(3):
            plot_all_alignments_in_direction(i, dsets)

        plt.show()
    finally:
        nplab.close_current_datafile()

