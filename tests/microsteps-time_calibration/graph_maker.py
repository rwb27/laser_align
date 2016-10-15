import yaml
import laser_align.data_io as d
import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt

with open('../../../data/microsteps-time_calibration/motor_speed.yaml',
          'r') as f:
    times_data = yaml.load(f)

series_dict = {}
full_data = []
l = ['x', 'y', 'z']
for i in range(3):
    #print times_data[i]
    times_data[i] = times_data[i][0, ...]
    times_data[i][:, 0] = np.abs(times_data[i][:, 0])
    series = []
    for each_microstep in np.nditer(np.unique(times_data[i][:, 0])):
        thing = times_data[i][np.where(times_data[i][:, 0] == each_microstep)][:, 1]
        series.append([each_microstep, np.mean(thing), 0, np.std(thing, ddof=1)])
    full_data.append(series)
    series = np.array(series)
    #print s.linregress(series[:, 0], series[:, 1])
    series_dict[l[i]] = (series, ('nopref', 'none'))

data = np.array(full_data)[0]
x = data[:, 0]
y = data[:, 1]

slope, intercept, r, prob2, see = s.linregress(x, y)
mx = x.mean()
sx2 = ((x - mx) ** 2).sum()
sd_intercept = see * np.sqrt(1. / len(x) + mx * mx / sx2)
sd_slope = see * np.sqrt(1. / sx2)

print intercept, sd_intercept, slope, sd_slope, r

d.plot_prettify(series_dict, 'Motor runtime against number of microsteps',
                'No. of '
                                                           'microsteps, m',
                'Time taken t/s', x_log=True,
                y_log=True, output='none', cust_x_lims=(50, 10**5))

plt.plot(x, intercept + x*slope, 'k:')
plt.show()