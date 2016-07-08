"""data_output.py
Contains functions to output data appropriately."""

import os
from datetime import datetime
import numpy as np


def save_dat(arr, pref, comment=False, location='../Data', time=True,
             append=True):
    """Saves a .d file to disk.
    :param arr: Numpy array to store.
    :param pref: Filename prefix, e.g. 'autofocus'.
    :param comment: Optional string to append to file.
    :param location: The folder in which data files are
    :param time: Whether or not to include time in the filename.
    :param append: Appends data to file if true, else overwrites previous.
    :return The time of file write, formatted as a string."""

    # USAGE NOTE: To append to a single file at multiple times, use this
    # function once under default settings, and for all subsequent appends
    # keep the same 'pref', but set 'comment' equal to the time returned on the
    # first run of the function, and 'time'=False. Note this can only be
    # done on the same day - to append to a previous day's data copy the
    # .d file into a new folder for the current date.

    dir_path = '{}/{}'.format(location, datetime.now().date())
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_name = pref
    if comment:
        file_name = '{}-{}'.format(file_name, comment)
    if time:
        time = datetime.now().time().strftime('%H.%M.%S')
        file_name = '{}-{}'.format(file_name, time)

    open_mode = 'a' if append else 'w'
    with open('{}/{}.d'.format(dir_path, file_name), open_mode) as f_handle:
        np.savetxt(f_handle, arr)

    return time

if __name__ == '__main__':
    the_time = save_dat(np.array([878]), 'ddd')
    save_dat(np.array([872382]), 'ddd', comment=the_time, time=False)
