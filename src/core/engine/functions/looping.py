import numpy as np


def fromto(a, b):
    """Inclusive Range"""
    return (float(x) for x in np.arange(a, b + 1))


def FROMTO(a, b):
    """Inclusive Range"""
    return f"fromto({a}, {b})"


def triangular(a, b):
    """Triangular Range: from |a-b| to a+b"""
    return fromto(abs(a - b), a + b)


def TRIANGULAR(a, b):
    """Triangular Range: from |a-b| to a+b"""
    return f"triangular({a}, {b})"


def intersection(*args):
    """Intersect ranges"""
    sets = [set(arg) for arg in args]
    return tuple(set.intersection(*sets))


def INTERSECTION(*args):
    """Intersect ranges"""
    return f"intersection({', '.join(args)})"


def projection(a):
    """Range from -a to a (inclusive)"""
    return fromto(-a, a)


def PROJECTION(a):
    """Range from -a to a (inclusive)"""
    return f"projection({a})"


def VALUE(a):
    """Just single value"""
    return f"({a},)"
