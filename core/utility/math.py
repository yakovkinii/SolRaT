from functools import lru_cache


def m1p(a):
    """
    real (-1)^a
    a: integer
    """
    return 1 - 2 * (a % 2)


@lru_cache(maxsize=None)
def fact(a):
    """
    int a!
    0!=1
    """
    result = 1
    for i in range(1, a + 1):
        result *= i
    return result


def fact2(a):
    """
    int (a/2)!
    a is doubled integer
    Input integrity assumed
    """
    return fact(a // 2)
