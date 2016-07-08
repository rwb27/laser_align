#!/usr/bin/env python

"""data_output.py
This script contains classes and function to ease data output from the
microscope. Adapted from the script by James Sharkey, 2015, which was used for
the paper in Review of Scientific Instruments titled: A one-piece 3D printed
flexure translation stage for open-source microscopy."""

import datetime
import h5py
import numpy as np
import simplejson as json
from jsmin import jsmin


class Datafile:
    _DEFAULT_FILE = "datafile"

    def __init__(self, filename=None):
        """A class to manage a hdf5 datafile.
        :param filename: If file name is specified, it should be a string
        ending in .hdf5, otherwise a filename is automatically generated. If
        no filename is explicitly specified, do not assume that just because a
        Datafile object exists, a corresponding file exists on disk. It may
        not exist until a group is created and data added. This may hide
        read/write privilege errors until late in execution."""

        today = datetime.date.today()
        self._date = today.strftime('%Y%m%d')

        if filename is None:  # If not explicitly asked for a datafile:
            self._filename = self._DEFAULT_FILE + "_" + self._date + ".hdf5"
            self._datafile = None  # Don't make one just yet
        else:
            self._datafile = h5py.File(filename, 'a')

    def new_group(self, group, attrs=None):
        """Create a new group with 'group_xxx' as the name, and returns it.
        - A timestamp is automatically added.
        - Use add_data(...) to create a dataset; since this manages
        attributes correctly.
        - (May overflow after 999 groups of same name.)

        :param group: The name of the group.
        :param attrs: A dictionary of attributes to add to the dataset, in the
        format {'attribute_name': value}."""

        # If not asked for datafile, but do need one, make one using the
        # filename generated.
        if self._datafile is None:
            self._datafile = h5py.File(self._filename, 'a')
        keys = self._datafile.keys()
        n = 0
        while group + "_%03d" % n in keys:
            n += 1
        group_path = group + "_%03d" % n
        g = self._datafile.create_group(group_path)

        self.add_attr(g, attrs)
        return g

    def add_data(self, data_array, group_object, dataset, attrs=None):
        """Given a datafile group object, create a dataset inside it from an
        array. May overflow after 99999 datasets of same name.
        :param data_array: An array-like object containing the dataset.
        :param group_object: The group object to which the dataset is to be
        added.
        :param dataset: Used to name the dataset, with a number appended.
        :param attrs: A dictionary of attributes to add to the dataset, in the
        format {'attribute_name': value}."""

        data_array = np.array(data_array)
        keys = group_object.keys()
        n = 0
        while dataset + "%05d" % n in keys:
            n += 1

        dataset += "%05d" % n
        d_set = group_object.create_dataset(dataset, data=data_array)

        self.add_attr(d_set, attrs)
        self._datafile.flush()

    @staticmethod
    def add_attr(data_obj, attr_dict=None):
        """Used to add attributes to a data object."""
        if attr_dict is None:
            attr_dict = {}
        # Add a timestamp to every data object.
        if 'timestamp' not in attr_dict.keys():
            attr_dict['timestamp'] = datetime.datetime.now().isoformat()
        for attribute in attr_dict:
            data_obj.attrs.create(attribute, attr_dict[attribute])

    def _close(self):
        """Close the file object and clean up. Called on deletion, do not call
        explicitly."""
        if self._datafile is not None:
            self._datafile.flush()
            self._datafile.close()

    def __del__(self):
        self._close()


def config_read(json_file):
    """Parses a JSON config file with comments and returns the output
    dictionary.
    :param json_file: The JSON file name, ending in .json."""
    with open(json_file) as js_file:
        no_comments = jsmin(js_file.read(), quote_chars="'\"`")
    return json.loads(no_comments)
