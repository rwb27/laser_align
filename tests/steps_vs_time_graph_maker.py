import yaml
import data_io as d
import numpy as np
import scipy.stats as s

with open('./motor_speed.yaml', 'r') as f:
    times_data = yaml.load(f)

series_dict = {}
for i in range(3):
    print times_data[i]
    times_data[i] = times_data[i][0, ...]
    times_data[i][:, 0] = np.abs(times_data[i][:, 0])
    series = []
    for each_microstep in np.nditer(np.unique(times_data[i][:, 0])):
        thing = times_data[i][np.where(times_data[i][:, 0] == each_microstep)][:, 1]
        series.append([each_microstep, np.mean(thing), 0, np.std(thing, ddof=1)])
    series = np.array(series)
    print s.linregress(series[:, 0], series[:, 1])
    series_dict[i] = (series, ('nopref', 'none'))

d.plot_prettify(series_dict, 'plot', 'microsteps', 'time', x_log=True,
                y_log=True)