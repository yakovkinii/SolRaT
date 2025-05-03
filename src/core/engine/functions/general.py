import math
from functools import lru_cache


def ensure_positive_if_zero(a: float):
    if a == -0.0:
        return 0.0
    else:
        return a


def half_int_to_str(a: float):
    """
    Converts half-integer floats like 0, 0.5, 1, ... to string in a stable manner
    (eliminates float artifacts)
    """
    if a % 1 not in [0, 0.0, 0.5]:
        raise ValueError(f"Expected half-integer, got {a}.")
    return str(ensure_positive_if_zero(round(float(a), 1)))


def m1p(a):
    """
    real (-1)^a
    a: integer
    """
    return 1 - 2 * (a % 2)


def delta(a, b):
    """
    real delta_ab
    a, b: integer
    """
    return 1 if a == b else 0


@lru_cache(maxsize=None)
def fact(a):
    """
    int a!
    0!=1
    """
    return math.factorial(a)
    # result = 1
    # for i in range(1, int(a) + 1):
    #     result *= i
    # return result


def fact2(a):
    """
    int (a/2)!
    a is doubled integer
    Input integrity assumed
    """
    return fact(int(a) // 2)


@lru_cache(maxsize=None)
def n_proj(*args):
    """
    n -> 2 n + 1. If multiple arguments - multiply
    """
    result = 1
    for arg in args:
        result *= 2 * arg + 1
    return result
