import numpy as np
import _experiments as _exp

#def consecutive_w_latest(results, num_per_parabola, max_step, n, axis_index):
#    """Get the maximum number of last rows from results, i.e. between
#    num_per_parabola and n rows, such that those rows have readings that
#    are consecutive (each readings differs from its neighbour by <=
#    max_step) and only one axis is varying. NOTE: n is the number of
#    extra results to get on each side of the set just measured."""
#
#    # Build the array of positions to visit by initially getting the
#    # last positions measured, and then sorting the array by the varying
#    # axis and looking for consecutive positions with the same values of
#    # the other position co-ordinates and gain. There is no need to
#    # check for uniqueness because this set has been measured at once.
#    last_rows = _check_sliced_results(
#        results, num_per_parabola, axis_index, check_unique=False)[:, 1:4]
#
#
#    # Get consecutive positions with the same value of gain.
#    return find_consecutive(results, last_rows, axis_index,
#                            max_step, n, 70)


def filter_gain(results, gain):
    """Return the results array only containing the rows with same gain
    specified."""
    return results[np.where(results[:, -1] == gain)]


def filter_unchanged_axes(results, axis_index, example_position):
    """Return the part of the array where the values for the unchanged
    position axes are those in example_position (an example position array
    where only axis_index - 1 is allowed to vary)."""

    for i in xrange(1, 4):
        if i != axis_index:
            results = results[np.where(results[:, i] == example_position[
                i - 1]), :][0]

    return results


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
    print last_rows
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


def delete_old_values(results):
    """Delete values older than those taken > DRIFT_TIME ago."""
    now_elapsed = _exp.elapsed(self.scope.start)
    return np.delete(results, np.where(now_elapsed - results[:, 0] >
                                       self.DRIFT_TIME), axis=0)


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


def find_consecutive(same_gain, last_rows, axis_index, max_step, n, gain):
    """Find n consecutive positions on on either side of the positions
    array, in the sorted same_gain array. Consecutive means they differ
    from their neighbours by <= max_step."""

    positions = last_rows[:, 1:4]

    sorted = sort_array(same_gain, axis_index)

    # Find the last measured positions in this sorted array. As they
    # were measured with the same gain and one axis changing, with the
    # same value of other positions, they should appear together as a
    # chunk.
    # Get the most recent (hence the last element of the np.where array,
    # as the sorted array is sorted by time) index of the positions measured
    # first and last respectively.
    min_position = np.where((sorted[:, 1:4] == positions[0, :]).all(
        axis=1))[-1]
    max_position = np.where((sorted[:, 1:4] == positions[-1, :]).all(
        axis=1))[-1]


    rows_to_check = last_rows
    print "rows to check", rows_to_check

    below_min_num = 0
    while min_row_index - 1 >= 0 and below_min_num <= n:
        if 0 <= sorted[min_row_index, axis_index] - \
                sorted[min_row_index - 1, axis_index] <= max_step:
            np.insert(rows_to_check, 0, sorted[min_row_index - 1, :],
                      axis=0)
            below_min_num -= 1
            min_row_index -= 1

    above_max_num = 0
    while max_row_index + 1 < sorted[:, 0].size and above_max_num <= n:
        if 0 <= sorted[max_row_index + 1, axis_index] - \
                sorted[max_row_index, axis_index] <= max_step:
            np.vstack((rows_to_check, sorted[max_row_index + 1, :]))
            above_max_num += 1
            max_row_index += 1
    return rows_to_check


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
results = np.vstack((results, np.array([1,2,1,4,12,3,4,5,60])))
print results


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


print get_most_recent(results, 3)
