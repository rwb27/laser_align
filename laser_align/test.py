import numpy as np
import _experiments as _exp


def get_proc_consec(results, axis_index, max_step):
    """From an array of results, select the appropriate ones to check for a
    parabola later.
    :param results: The results array, in the forat with column titles: time
    elapsed, x, y, z, brightness, error, number per average, delay per mmt,
    gain.
    :param axis_index: The index of the position axis in the array that was
    just measured - only values varying along this axis are considered.
    :param max_step: The maximum step size for measurements to be considered
    consecutive.
    :return: The measurements that the consecutive to the ones just taken."""

    # This function must be applied just after a series of measurements,
    # which are sorted by time.
    last_row = results[-1, :]
    example_pos = last_row[1:4]
    gain = last_row[-1]

    # To get the results ready for processing, the algorithm is:
    # 1) Only keep rows with the same values for the unchanged position axes
    #    as the last mmt set.
    # 2) Only keep rows with the same gain as the last mmt set.
    # 3) Sort the array by time and positions.
    # 4) Delete values older than DRIFT_TIME.
    # 5) Get the most recent versions of values if multiple have been taken
    #    at the same point.
    # 6) Find all values consecutive to the ones taken in the last set,
    #    along the varying axis.
    results = filter_unchanged_axes(results, axis_index, example_pos)
    print results
    results = filter_gain(results, gain)
    print results
    results = sort_array(results, axis_index)
    print results
    results = delete_old_values(results)
    print results
    results = get_most_recent(results, axis_index)
    print results
    results = find_consecutive(results, axis_index, max_step)

    return results


def filter_unchanged_axes(results, axis_index, example_position):
    """Return the part of the array where the values for the unchanged
    position axes are those in example_position (an example position array
    where only axis_index - 1 is allowed to vary)."""

    for i in xrange(1, 4):
        if i != axis_index:
            results = results[np.where(results[:, i] == example_position[
                i - 1]), :][0]

    return results


def filter_gain(results, gain):
    """Return the results array only containing the rows with same gain
    specified."""
    return results[np.where(results[:, -1] == gain)]


def sort_array(results, axis_index):
    """Sort an array by the first 4 indices (time, 3-displacement).
    :param results: Results array filtered by one specific gain value.
    :param axis_index: The index of the position axis that varies."""

    # Get the order in which to sort the array. Sort first by the axes
    # that do not change. This should result in [0, axis_index,
    # other_pos_axes] at the end. So, sorting occurs by the unchanged
    # position axes first, then by the changing position, then by time.
    sort_by = []
    for axis in xrange(4):
        if axis != axis_index:
            sort_by.append(axis)
    sort_by.insert(1, axis_index)

    # This method sorts by the last element in the tuple first.
    row_order = np.lexsort((results[:, sort_by[0]],
                            results[:, sort_by[1]],
                            results[:, sort_by[2]],
                            results[:, sort_by[3]]))
    return results[row_order, :]


def delete_old_values(results):
    """Delete values older than those taken > DRIFT_TIME ago."""
    now_elapsed = _exp.elapsed(self.scope.start)
    deleted = np.delete(results, np.where(now_elapsed - results[:, 0] >
                                          self.DRIFT_TIME), axis=0)
    if deleted.size > 0:
        return deleted
    else:
        raise ValueError


def get_most_recent(results, axis_index):
    """Find the most recent value of each measurement in results, assuming
    it has already been sorted, filtered for gain, old values deleted,
    and filtered by position axes that do not change."""
    recent_array = []
    for position in np.nditer(np.unique(results[:, axis_index])):
        temp = results[np.where(results[:, axis_index] == position)]
        temp = temp[np.where(temp[:, 0] == np.max(temp[:, 0])), :][0][0]
        recent_array.append(temp)
    return np.array(recent_array)


def find_consecutive(results, axis_index, max_step):
    """Find n consecutive positions on on either side of the positions
    array, in the sorted results array. Consecutive means they differ
    from their neighbours by <= max_step."""

    # Readings are taken as not consectuive when they differ from their
    # neighbours by > max_step. The code below gives the indices to split
    # the results array along, into different sets of measurements.
    split_along = np.where(np.absolute(np.ediff1d(results[:, axis_index])) >
                           max_step)[0] + 1
    return np.split(results, split_along)


def _check_sliced_results(results, slice_size, axis_index=None,
                          check_unique=True):
    """Ensure the other position axes haven't changed, and the axis in
    question has all different positions. Return the appropriately
    sliced results.
    :param results: The results array/list.
    :param axis_index: The index of the position that is varying.
    :param slice_size: The number of the last rows of the results array to
    use."""
    last_rows = np.array(results[-slice_size:])
    assert last_rows.shape[0] == slice_size

    if check_unique:
        other_indices = []
        for ind in [1, 2, 3]:
            if ind != axis_index:
                other_indices.append(ind)
        assert [np.unique(last_rows[:, other_index]).size == 1 for
                other_index in other_indices]

        assert np.unique(last_rows[:, axis_index]).size == slice_size

    return last_rows


times = np.random.random(12)*1000
x = np.ones(3)
x = np.concatenate((x, x*2, x*3, x*4))
y = np.ones(12)
z = np.ones(1)
z = np.concatenate((z, z*2, z*3, z*4, z*5, z*6, z*7, z*8, z*9, z*10, z*11,
                    z*12))
brightness = np.random.random(12)*1023
error = np.random.random(12)
number = np.ones(12)*100
delay = np.zeros(12)
gain = np.ones(6)
gain = np.concatenate((gain*70, gain*60))
results = np.vstack((times, x, y, z, brightness, error, number, delay, gain)).T
results = np.vstack((results, np.array([1000,2,1,4,12,3,4,5,60])))
print results
print get_proc_consec(results, 1, .8)
