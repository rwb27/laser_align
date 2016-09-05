import h5py
import sensor_scope.data_io as d
import numpy as np
import matplotlib.pyplot as plt


df = h5py.File(r'C:\Users\a-amb\OneDrive - University Of Cambridge\2016-09-02-'
               r'timed-no-averaging.h5', 'r')
data = df["TimedMeasurements/TimedMeasurements_0/brightness_final"][:, ...]

thing = d.series_maker('time', x=data[:, 0], y=data[:, 4], xerr=None,
                       yerr=data[:, 5], line_format='none')
d.plot_prettify(thing, 'brightness_vs_time', 'time/s', 'brightness/AU',
                output='show')


def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

yMA = moving_average(data[:, 4], 500)
plt.plot(data[:, 0][len(data[:, 0])-len(yMA):], yMA)
