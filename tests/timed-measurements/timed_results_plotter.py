import h5py
import laser_align.data_io as d
import numpy as np
import matplotlib.pyplot as plt


df = h5py.File(r'C:\Users\Abhishek\OneDrive - University Of '
               r'Cambridge\data\timed-measurements\2016-09-02-timed-no-averaging.h5', 'r')
data = df["TimedMeasurements/TimedMeasurements_0/brightness_final"][:, ...]

#plt.hist(data[:, 4], 100)

#ax = plt.gca()
#ax.tick_params(direction='out', labelsize=12)
#plt.grid(True)
#plt.xlabel('Intensity/AU', fontsize=14, fontweight='bold')
#plt.ylabel('Count', fontsize=14, fontweight='bold')
##plt.title('Histogram', fontsize=16, fontweight='bold', y=1.05)
#plt.ticklabel_format(useOffset=False)
#
#plt.show()

thing = d.series_maker('time', x=data[:, 0], y=data[:, 4], xerr=None,
                       yerr=data[:, 5], marker_format='+', line_format='none')
d.plot_prettify(thing, 'Intensity as a function of time', 'Time/s',
                'Intensity/AU',
                output='none')


def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

yMA = moving_average(data[:, 4], 100)
plt.plot(data[:, 0][len(data[:, 0])-len(yMA):], yMA)
plt.show()