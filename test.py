def check_defaults(list_of_vars, config_dict, list_of_keys):
    """Check if each variable in list_of_vars is None, and change it to the
    corresponding value in config_dict if so.
    :param list_of_vars: List of variables to check.
    :param config_dict: Configuuration dictionary.
    :param list_of_keys: List of respective key names.
    :return: The modified list_of_vars."""
    for i in range(len(list_of_vars)):
        if list_of_vars[i] is None:
            list_of_vars[i] = config_dict[list_of_keys[i]]

    return list_of_vars


print check_defaults([None, 'thing', None, 'banana', 3], {'one': 'first',
                                                         'two':
    'second', 'three': 'third', 'four': 'fourth', 'five': 'fifth'}, [
    'one', 'two', 'three', 'four', 'five'])