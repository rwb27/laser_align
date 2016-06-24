"""Contains base-level functions that are required for the others to run."""


def factors(num):
    """Returns the factors of a number."""
    factor_list = list(reduce(list.__add__, ([j, num // j] for j in range(
        1, int(num ** 0.5) + 1) if num % j == 0)))
    factor_list.sort()
    return factor_list

