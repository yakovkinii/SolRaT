import numpy as np
from numpy import abs, sqrt

from core.utility.math import fact2


def _w3j(j1_doubled, j2_doubled, j3_doubled, m1_doubled, m2_doubled, m3_doubled):
    """
    float Wigner 3J symbol where all arguments are doubled to be integer

    j1_doubled = 2 * J1: int
    j2_doubled = 2 * J2: int
    j3_doubled = 2 * J3: int
    m1_doubled = 2 * M1: int
    m2_doubled = 2 * M2: int
    m3_doubled = 2 * M3: int

    Reference: Appendix A1

    ( J1 J2 J3 )
    ( M1 M2 M3 )
    """
    if m1_doubled + m2_doubled + m3_doubled != 0:
        return 0.0
    if (
        (abs(m1_doubled) > j1_doubled)
        or (abs(m2_doubled) > j2_doubled)
        or (abs(m3_doubled) > j3_doubled)
    ):
        return 0.0
    a = j1_doubled + j2_doubled
    if j3_doubled > a:
        return 0.0
    b = j1_doubled - j2_doubled
    if j3_doubled < abs(b):
        return 0.0
    j_sum = j3_doubled + a
    c = j1_doubled - m1_doubled
    d = j2_doubled - m2_doubled
    if (j_sum % 2 != 0) or (c % 2 != 0) or (d % 2 != 0):
        return 0.0
    e = j3_doubled - j2_doubled + m1_doubled
    f = j3_doubled - j1_doubled - m2_doubled
    z_min = max(0, -e, -f)
    g = a - j3_doubled
    h = j2_doubled + m2_doubled
    z_max = min(g, h, c)
    result = 0.0
    for z in range(z_min, z_max + 1, 2):
        denominator = (
            fact2(z)
            * fact2(g - z)
            * fact2(c - z)
            * fact2(h - z)
            * fact2(e + z)
            * fact2(f + z)
        )
        if z % 4 != 0:
            denominator = -denominator
        result += 1 / denominator
    cc1 = fact2(g) * fact2(j3_doubled + b) * fact2(j3_doubled - b) / fact2(j_sum + 2)
    cc2 = (
        fact2(j1_doubled + m1_doubled)
        * fact2(c)
        * fact2(h)
        * fact2(d)
        * fact2(j3_doubled - m3_doubled)
        * fact2(j3_doubled + m3_doubled)
    )
    result *= sqrt(cc1 * cc2)
    if (b - m3_doubled) % 4 != 0:
        result = -result
    return result


def _w6j(j1_doubled, j2_doubled, j3_doubled, l1_doubled, l2_doubled, l3_doubled):
    """
    float Wigner 6J symbol where all arguments are doubled to be integer

    j1_doubled = 2 * J1: int
    j2_doubled = 2 * J2: int
    j3_doubled = 2 * J3: int
    l1_doubled = 2 * L1: int
    l2_doubled = 2 * L2: int
    l3_doubled = 2 * L3: int

    Reference: Appendix A1

    { J1 J2 J3 }
    { L1 L2 L3 }
    """
    a = j1_doubled + j2_doubled
    b = j1_doubled - j2_doubled
    c = j1_doubled + l2_doubled
    d = j1_doubled - l2_doubled
    e = l1_doubled + j2_doubled
    f = l1_doubled - j2_doubled
    g = l1_doubled + l2_doubled
    h = l1_doubled - l2_doubled

    if (a < j3_doubled) or (c < l3_doubled) or (e < l3_doubled) or (g < j3_doubled):
        return 0.0
    if (
        (abs(b) > j3_doubled)
        or (abs(d) > l3_doubled)
        or (abs(f) > l3_doubled)
        or (abs(h) > j3_doubled)
    ):
        return 0.0

    sum_1 = a + j3_doubled
    sum_2 = c + l3_doubled
    sum_3 = e + l3_doubled
    sum_4 = g + j3_doubled

    if (sum_1 % 2 != 0) or (sum_2 % 2 != 0) or (sum_3 % 2 != 0) or (sum_4 % 2 != 0):
        return 0.0

    w_min = max(sum_1, sum_2, sum_3, sum_4)
    i = a + g
    j = j2_doubled + j3_doubled + l2_doubled + l3_doubled
    k = j3_doubled + j1_doubled + l3_doubled + l1_doubled
    w_max = min(i, j, k)

    result = 0.0
    for w in range(w_min, w_max + 1, 2):
        denominator = (
            fact2(w - sum_1)
            * fact2(w - sum_2)
            * fact2(w - sum_3)
            * fact2(w - sum_4)
            * fact2(i - w)
            * fact2(j - w)
            * fact2(k - w)
        )
        if w % 4 != 0:
            denominator = -denominator
        result += fact2(w + 2) / denominator

    theta1 = (
        fact2(a - j3_doubled)
        * fact2(j3_doubled + b)
        * fact2(j3_doubled - b)
        / fact2(sum_1 + 2)
    )
    theta2 = (
        fact2(c - l3_doubled)
        * fact2(l3_doubled + d)
        * fact2(l3_doubled - d)
        / fact2(sum_2 + 2)
    )
    theta3 = (
        fact2(e - l3_doubled)
        * fact2(l3_doubled + f)
        * fact2(l3_doubled - f)
        / fact2(sum_3 + 2)
    )
    theta4 = (
        fact2(g - j3_doubled)
        * fact2(j3_doubled + h)
        * fact2(j3_doubled - h)
        / fact2(sum_4 + 2)
    )
    result = result * sqrt(theta1 * theta2 * theta3 * theta4)
    return result


def _w9j(
    j1_doubled,
    j2_doubled,
    j3_doubled,
    j4_doubled,
    j5_doubled,
    j6_doubled,
    j7_doubled,
    j8_doubled,
    j9_doubled,
):
    """
    float Wigner 9J symbol where all arguments are doubled to be integer

    j1_doubled = 2 * J1: int
    j2_doubled = 2 * J2: int
    j3_doubled = 2 * J3: int
    j4_doubled = 2 * J4: int
    j5_doubled = 2 * J5: int
    j6_doubled = 2 * J6: int
    j7_doubled = 2 * J7: int
    j8_doubled = 2 * J8: int
    j9_doubled = 2 * J9: int

    Reference: Appendix A1

    { J1 J2 J3 }
    { J4 J5 J6 }
    { J7 J8 J9 }
    """
    k_min = max(
        abs(j1_doubled - j9_doubled),
        abs(j4_doubled - j8_doubled),
        abs(j2_doubled - j6_doubled),
    )

    k_max = min(
        abs(j1_doubled + j9_doubled),
        abs(j4_doubled + j8_doubled),
        abs(j2_doubled + j6_doubled),
    )
    result = 0
    for k in range(k_min, k_max + 1, 2):
        s = -1 if k % 2 != 0 else 1
        x1 = w6j_doubled(j1_doubled, j9_doubled, k, j8_doubled, j4_doubled, j7_doubled)
        x2 = w6j_doubled(j2_doubled, j6_doubled, k, j4_doubled, j8_doubled, j5_doubled)
        x3 = w6j_doubled(j1_doubled, j9_doubled, k, j6_doubled, j2_doubled, j3_doubled)
        result += s * x1 * x2 * x3 * (k + 1)
    return result


# TODO: pre-calculating first can improve performance:
# TODO: pre-calculate for common inputs -> map -> vectorize -> wrap: apply map, then fill NA by calling explicitly
w3j_doubled = np.vectorize(_w3j)
w6j_doubled = np.vectorize(_w6j)
w9j_doubled = np.vectorize(_w9j)


def w6j(j1, j2, j3, l1, l2, l3):
    return w6j_doubled(j1 * 2, j2 * 2, j3 * 2, l1 * 2, l2 * 2, l3 * 2)
