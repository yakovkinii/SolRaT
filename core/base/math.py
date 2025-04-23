from functools import lru_cache


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


δ = delta


@lru_cache(maxsize=None)
def fact(a):
    """
    int a!
    0!=1
    """
    result = 1
    for i in range(1, int(a) + 1):
        result *= i
    return result


def fact2(a):
    """
    int (a/2)!
    a is doubled integer
    Input integrity assumed
    """
    return fact(int(a) // 2)
