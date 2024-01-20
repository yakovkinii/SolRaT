import numpy as np


def range_inclusive(a, b=None, c=1, /):
    """Inclusive Range"""
    if b is None:
        return np.arange(a + 1)
    return np.arange(a, b + 1, c)
