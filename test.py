import numpy as np


def _get_minimum(col):
    min = np.min(col)
    if min < 0:
        col = col + abs(min)
        return col


def array_map(x):
    return np.array(list(map(_get_minimum, x)))


array = np.array([[-1, -5], [5, 6]])
