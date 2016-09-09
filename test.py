import numpy as np


def last_n_rows(results, num_per_parabola, max_step, n):
    """Get the maximum number of last rows from results, i.e. between
    num_per_parabola and n rows, such that those rows have readings that
    are consecutive (each readings differs from its neighbour by <=
    max_step) and only one axis is varying. NOTE: n  is the number of
    extra results to get on each side of the set just measured."""

    # The position axis that is taken to vary is the one that changes
    # between the last and second last results before sorting.
    changed = results[-1, 1:4] - results[-2, 1:4]
    print changed
    axis_index = int(np.where(changed != 0)[0]) + 1

    # Build the array of positions to visit by initially getting the
    # last positions measured, and then sorting the array by the varying
    # axis and looking for consecutive positions with the same values of
    # the other position co-ordinates and gain. There is no need to
    # check for uniqueness because this set has been measured at once.
    positions = _check_sliced_results(
        results, num_per_parabola, axis_index, check_unique=False)[:, 1:4]
    # Get consecutive positions with the same value of gain.
    return find_consecutive(results, positions, axis_index,
                            max_step, n, 70)


def get_most_recent(results, position):
    """Searches for the position array in the results, and returns that
    row. If the same position is found multiple times, the most recent
    position is returned. Gain is also matched with current gain."""
    # Get rows with the matched position
    match = get_matched_rows(results, position)

    # Get the results where the gain is the same as the current gain. A
    # single row must be returned, so find the most recent reading for
    # that position if there are multiple.
    return match[np.where(match[:, 0] == np.max(match[:, 0]))]


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


def find_consecutive(results, positions, axis_index, max_step, n, gain):
    """Find n consecutive positions on on either side of the positions
    array, in the sorted results array. Consecutive means they differ
    from their neighbours by <= max_step."""

    same_gain = results[np.where(results[:, -1] == gain)]
    print "positions", positions
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
    row_order = np.lexsort((same_gain[:, sort_by[0]],
                            same_gain[:, sort_by[1]],
                            same_gain[:, sort_by[2]],
                            same_gain[:, sort_by[3]]))
    sorted = same_gain[row_order, :]

    # Find the last measured positions in this sorted array. As they
    # were measured with the same gain and one axis changing, with the
    # same value of other positions, they should appear together as a
    # chunk. Find the consecutive positions.
    min_position = positions[np.where(positions[:, axis_index - 1] == np.min(
        positions[:, axis_index - 1])), :][0]
    max_position = positions[np.where(positions[:, axis_index - 1] == np.max(
        positions[:, axis_index - 1])), :][0]
    print min_position

    # Get the indices of the most recent values of the min and max
    # positions that were measured. Look above and below these.
    min_row_index = np.max(np.where((sorted[:, 0:] == get_most_recent(
        sorted, min_position)).all(axis=1)))
    max_row_index = np.max(np.where((sorted[:, 0:] == get_most_recent(
        sorted, max_position)).all(axis=1)))

    rows_to_check = sorted[min_row_index: max_row_index + 1, :]


    below_min_num = 0
    while min_row_index - 1 >= 0 and below_min_num <= n:
        if 0 <= sorted[min_row_index, axis_index] - \
                sorted[min_row_index - 1, axis_index] <= max_step:
            np.insert(rows_to_check, 0, sorted[min_row_index - 1, :],
                      axis=0)
            below_min_num -= 1

    above_max_num = 0
    while max_row_index + 1 <= sorted[:, 0].size and above_max_num <= n:
        if 0 <= sorted[max_row_index + 1, axis_index] - \
                sorted[max_row_index, axis_index] <= max_step:
            np.vstack((rows_to_check, sorted[max_row_index + 1, :]))
            above_max_num += 1
    return rows_to_check


def get_matched_rows(results, position):
    print results, position
    print results[np.where((results[:, 1:4] == position).all(axis=1)), :]
    return results[np.where((results[:, 1:4] == position).all(axis=1)), :]


times = np.random.random(12)*100
x = np.ones(3)
x = np.concatenate((x, x*2, x*3, x*4))
y = np.ones(4)
y = np.concatenate((y, y*2, y*3))
z = np.ones(1)
z = np.concatenate((z, z*2, z*3, z*4, z*5, z*6, z*7, z*8, z*9, z*10, z*11,
                    z*12))
brightness = np.random.random(12)*1023
error = np.random.random(12)
number = np.ones(12)*100
delay = np.zeros(12)
gain = np.ones(12)*70
results = np.vstack((times, x, y, z, brightness, error, number, delay, gain)).T
print last_n_rows(results, 7, 1, 4)

