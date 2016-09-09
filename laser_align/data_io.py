#!/usr/bin/env python

"""data_io.py
Contains functions for datafile processing and IO."""

import numpy as np
import yaml
from matplotlib import pyplot as plt


# Data input from YAML config files.
def yaml_read(yaml_file):
    """Parses a YAML file and returns the resulting dictionary.
    :param yaml_file: the YAML file path, ending in .yaml."""
    with open(yaml_file, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict


def make_dict(config, **kwargs):
    """Read the dictionary, or create one from a YAML file, and replace any
    optionally specified named parameters by those from kwargs. Returns the
    modified dictionary.
    :param config: Either a string specifying a path to the config file,
    or the dictionary of config parameters."""

    if type(config) is str:
        # Read default values from dictionary.
        config = yaml_read(config)
    elif type(config) is not dict:
        raise TypeError('Variable \'config\' is neither a string nor a '
                        'dictionary.')
    for key in kwargs:
        # Only existing parameters can be changed.
        assert key in config.keys(), "The key \'{}\' is not present in the " \
                                     "file. Ensure it has been typed " \
                                     "correctly.".format(key)
        config[key] = kwargs[key]

    return config


def sub_dict(main_dict, subset_keys=None, extra_entries=None):
    """Slice specific keys of a dictionary, add extra entries if specified,
    and return the new dictionary.
    :param main_dict: The main dictionary to slice.
    :param subset_keys: The tuple of keys in main_dict to use. If None,
    all keys are used.
    :param extra_entries: A dictionary of extra entries to add to slice
    dictionary."""
    subset_dict = {}
    if subset_keys is not None:
        for key in subset_keys:
            # Own KeyError will be raised if key does not exist in main_dict
            subset_dict[key] = main_dict[key]
    else:
        # subset_keys is None
        subset_dict = main_dict

    if extra_entries is not None:
        for key in extra_entries:
            subset_dict[key] = extra_entries[key]

    return subset_dict


def make_group(sc_exp_obj, group=None, group_name='Group'):
    """If 'group' is None, create a new group in the main datafile with name
    'group_name', else if 'group' is HDF5 group object, refer to that group.
    :param sc_exp_obj: ScopeExp object
    :param group: Either None or refers to a group object.
    :param group_name: The name of the group to be created, if group is None.
    :return: The created group object."""
    if group is None:
        sc_exp_obj.gr = sc_exp_obj.create_data_group(
            group_name, attrs=sc_exp_obj.attrs)
    else:
        sc_exp_obj.gr = group
    return sc_exp_obj.gr


# Graphing wrappers.
def plot_prettify(series_dict, title, x_title, y_title, output='show',
                  x_log=False, y_log=False, cust_x_lims=None, cust_y_lims=None,
                  x_sci=False, y_sci=False):
    """Plots a graph, possibly with error bars, and makes it look nice.
    :param series_dict: A dictionary of data series in the format
        {'series_1_name': ([x_data_col, y_data_col, x_err_col, y_err_col],
                           ('marker_format', 'line_form')),
                           ...}
    where each dictionary value is a tuple with an array as the first element
    and the marker and line preferences as the second element. Set every entry
    in an error column to 0 if it has no errors.
    :param title: Graph title string.
    :param x_title: X axis title string.
    :param x_log: Set x axis to log scale if True.
    :param x_sci: Set x axis scale to scientific.
    :param cust_x_lims: Set custom x-axis limits by passing in a (min,
    max) tuple. None means custom values retained.
    :param y_title, y_log, cust_y_lims, y_sci: Analogous for y.
    :param output: Graph output format. 'show' to output to screen, 'save'
    to save to current directory and 'none' to do nothing."""
    handles = []
    labels = []
    for series in series_dict:
        x = series_dict[series][0][:, 0]
        y = series_dict[series][0][:, 1]
        xerrs = _arr_to_one_none(series_dict[series][0][:, 2])
        yerrs = _arr_to_one_none(series_dict[series][0][:, 3])

        # Marker and line style codes are 'nopref' for default values; see
        # matplotlib for other possibilities.
        marker_format = series_dict[series][1][0]
        line_format = series_dict[series][1][1]

        if marker_format == 'nopref' and line_format == 'nopref':
            handles.append(plt.errorbar(x, y, xerr=xerrs, yerr=yerrs,
                                        label=series))
        elif marker_format == 'nopref':
            handles.append(plt.errorbar(x, y, xerr=xerrs, yerr=yerrs,
                                        label=series, linestyle=line_format))
        elif line_format == 'nopref':
            handles.append(plt.errorbar(x, y, xerr=xerrs, yerr=yerrs,
                                        label=series, marker=marker_format))
        else:
            handles.append(plt.errorbar(x, y, xerr=xerrs, yerr=yerrs,
                                        label=series, linestyle=line_format,
                                        marker=marker_format))
        labels.append(series)

    ax = plt.gca()
    ax.tick_params(direction='out', labelsize=12)
    plt.grid(True)
    plt.xlabel(x_title, fontsize=14, fontweight='bold')
    plt.ylabel(y_title, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', y=1.05)
    plt.ticklabel_format(useOffset=False)
    if x_sci:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    if y_sci:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    if cust_x_lims is not None:
        plt.xlim(cust_x_lims)
    if cust_y_lims is not None:
        plt.ylim(cust_y_lims)

    # Display legend for multiple series.
    if len(series_dict) > 1:
        plt.legend(handles, labels, loc='best')

    if output == 'save':
        plt.savefig('{}.png'.format(title.replace(' ', '_')))
        plt.clf()
    elif output == 'show':
        plt.show()
    elif output != 'none':
        raise ValueError("The variable 'output' is invalid.")

    return


def series_maker(series_name, x, y, xerr=None, yerr=None, series_dict=None,
                 marker_format='nopref', line_format='nopref'):
    """Appends a data series to the series_dict dictionary with the data
    array correctly formatted.
    :param series_name: The string key for series_dict.
    :param x: x data 1D array.
    :param xerr: x errors column array.
    :param y, yerr: Likewise for y.
    :param marker_format: The string specifying the marker format. 'nopref'
    keeps it to system-chosen value; see marker codes for matplotlib to see
    other values this can take.
    :param line_format: As above for the series line style.
    :param series_dict: The dictionary of data to append to.
    :return: series_dict with the new series added."""

    x = np.array(x)
    y = np.array(y)

    if series_dict is None:
        series_dict = {}
    if xerr is None:
        xerr = np.zeros(x.size)
    if yerr is None:
        yerr = np.zeros(y.size)

    data_array = np.vstack((x, y, xerr, yerr)).T
    series_dict[series_name] = (data_array, (marker_format, line_format))
    return series_dict


def _arr_to_one_none(arr):
    """If all values in array are zero, return None, otherwise return the
    array.
    :param arr: The array."""
    for entry in np.nditer(arr):
        if entry != 0:
            return arr
    return None
