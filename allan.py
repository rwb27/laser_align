"""Use functions here to calculate the Allan variance of brightness vs time
measurements."""

from scipy.integrate import simps
import numpy as np
import h5py
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


df = h5py.File(r'C:\Users\a-amb\OneDrive - University Of '
               r'Cambridge\data\timed-measurements\2016-08-26-timed-weekend'
               r'-run.h5')
results = []
for tau in np.arange(20, 50000, 50):
    allan_dev = allan(df['TimedMeasurements/TimedMeasurements_0'
                         '/brightness_final'][:, 0],
                      df['TimedMeasurements/TimedMeasurements_0'
                         '/brightness_final'][:, 4], tau)
    results.append([tau, allan_dev])

results = np.array(results)
plt.plot(results[:, 0], results[:, 1])

df2 = h5py.File(r'C:\Users\a-amb\OneDrive - University Of '
                r'Cambridge\timed_mmts_test_remote2.h5')
for dataset in [0]:
    results2 = []
    for tau in np.arange(10, 1000, 10):
        allan_dev = allan(df2['TimedMeasurements/TimedMeasurements_{}/'
                              'brightness_final'.format(dataset)][:, 0],
                          df2['TimedMeasurements/TimedMeasurements_{}'
                              '/brightness_final'.format(dataset)][:, 4], tau)
        results2.append([tau, allan_dev])

    results2 = np.array(results2)
    plt.plot(results2[:, 0], results2[:, 1])

plt.xscale('log')
plt.yscale('log')
plt.show()
df.close()
