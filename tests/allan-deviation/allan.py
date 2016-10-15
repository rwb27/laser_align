"""Use functions here to calculate the Allan variance of brightness vs time
measurements."""

from scipy.integrate import simps
from scipy.stats import linregress
import numpy as np
import h5py
import laser_align.data_io as d
import matplotlib.pyplot as plt


def allan(t, x, tau):
    """Calculate Allan variance using Simpson's rule for numerical integration.
    :param t: The t-series as an array.
    :param x: The x-array.
    :param tau: The time chunks to divide the series into.
    :return: The data for the allan variance."""

    assert x.size == t.size, "Arrays are of different lengths."

    m = (np.max(t) - np.min(t)) / float(tau)
    mmts_per_tau = x.size / float(m)

    x_i = []
    for i in xrange(int(round(m))):
        x_s = x[np.floor(i * mmts_per_tau): np.floor((i+1) * mmts_per_tau + 1)]
        t_s = t[np.floor(i * mmts_per_tau): np.floor((i+1) * mmts_per_tau + 1)]
        integrand = simps(x_s, t_s)
        x_i.append(integrand / (np.max(t_s) - np.min(t_s)))

    variances = (np.ediff1d(np.array(x_i))/2) ** 2
    allan_dev = np.sqrt(np.mean(variances))

    return allan_dev


def get_dataset(file_path, path_strings):
    """Opens the HDF5 datafile file_path, obtains the 0th and 4th column of
    the datasets specified by path_strings (a list of strings) and returns
    the datasets."""
    df = h5py.File(file_path)
    results = []
    for path_string in path_strings:
        t_set = df[path_string][:, 0]
        x_set = df[path_string][:, 4]
        results.append(np.vstack((t_set, x_set)).T)
    df.close()

    return results

series_dict = {}
big_results = []
deets = [['2016-08-26-timed-weekend-run-averageover10.h5', 100, 32400, 5],
         ['2016-09-02-timed-no-averaging.h5', 2, 500, 1]]
for filename in deets:
    df = h5py.File(r'C:\Users\Abhishek\OneDrive - University Of '
                   r'Cambridge\data\timed-measurements\{}'.format(filename[0]))
    results = []
    for tau in np.arange(filename[1], filename[2], filename[3]):
        allan_dev = allan(df['TimedMeasurements/TimedMeasurements_0'
                             '/brightness_final'][:, 0],
                          df['TimedMeasurements/TimedMeasurements_0'
                             '/brightness_final'][:, 4], tau)

        results.append([tau, allan_dev])

    results = np.array(results)
    big_results.append(results)
    series_dict = d.series_maker(filename[0], results[:, 0], results[:, 1],
                                 series_dict=series_dict)
    df.close()

big_results[0] = np.log10(big_results[0])
plt.loglog(big_results[0][:, 0][100:], big_results[0][:, 1][100:])
plt.show()
plt.loglog(big_results[1][:, 0][:-440], big_results[1][:, 1][:-440])
plt.show()

x, y = big_results[1][:, 0][:-440], big_results[1][:, 1][:-440]

slope, intercept, r, prob2, see = linregress(x, y)
mx = x.mean()
sx2 = ((x - mx) ** 2).sum()
sd_intercept = see * np.sqrt(1. / len(x) + mx * mx / sx2)
sd_slope = see * np.sqrt(1. / sx2)

print slope, sd_slope
