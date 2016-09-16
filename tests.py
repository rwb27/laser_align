import scipy.stats as stat
import scipy.integrate as integrate
import numpy as np


def thresh_com(x, y):
    """Threshold the results at the half-range level and find the
    centre of mass of the peak of maximum intensity-width.
    :param x: The array of the last set of positions, sorted in
    ascending order.
    :param y: The brightness measurements corresponding to x.
    :return: The centre of mass in position and the thresholded
    brightness array."""
    thresh_level = np.mean([np.min(y), np.max(y)])
    thresh = stat.threshold(y, threshmin=thresh_level)
    normalisation = integrate.simps(thresh, x)
    numerator = integrate.simps(x * thresh, x)

    return numerator / normalisation, thresh

thing = np.array([0, 20,0,20,20,0,0,10])
thing2 = np.array(np.split(thing, np.where(thing == 0)[0]))
print thing2
for arr in thing2:
    print "arr",  arr
    print int(np.count_nonzero(arr))
    if arr.size == 0 or np.count_nonzero(arr) == 0:
        thing2 = np.delete(thing2, arr)
    else:
        if arr[0] != 0:
            arr = np.insert(arr, 0, [0])
            print arr
        if arr[-1] != 0:
            arr = np.append(arr, [0])
            print arr
print np.pad(thing2, (1, 1), mode='constant')
print thing2
