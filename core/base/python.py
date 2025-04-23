import numpy as np


def range_inclusive(a, b):
    """Inclusive Range"""
    return [x for x in np.arange(a, b + 1)]


def triangular(a, b):
    """Triangular Range: from |a-b| to a+b"""
    return range_inclusive(abs(a - b), a + b)


def intersection(a, b):
    """Intersect 2 ranges"""
    return [item for item in a if item in b]


def projection(a):
    """Range from -a to a (inclusive)"""
    return range_inclusive(-a, a)


def triangular_with_kr(a):
    """Triangular range with Kr, which is <=2"""
    return range_inclusive(max(a - 2, 0), a + 2)


def ensure_positive_if_zero(a: float):
    if a == -0.0:
        return 0.0
    else:
        return a


def half_int_to_str(a: float):
    """
    Converts half-integer floats like 0, 0.5, 1, ... to string in a stable manner (eliminates float artifacts)
    """
    if a % 1 not in [0, 0.0, 0.5]:
        raise ValueError(f"Expected half-integer, got {a}.")
    return str(ensure_positive_if_zero(round(float(a), 1)))
