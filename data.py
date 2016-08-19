#!/usr/bin/env python

"""data.py
Contains functions for datafile processing and IO."""

import yaml


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
    :param subset_keys: The list of keys in main_dict to use. If None,
    all keys are used.
    :param extra_entries: A dictionary of extra entries to add to slice
    dictionary."""
    subset_dict = {}
    if subset_keys is not None:
        for key in subset_keys:
            subset_dict[key] = main_dict[key]
    elif subset_keys is None:
        subset_dict = main_dict
    else:
        raise ValueError('subset_keys is invalid.')

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