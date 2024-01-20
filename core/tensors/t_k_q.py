import numpy as np
from numpy import cos, sin, exp

from core.utility.constant import sqrt2, sqrt3
from core.utility.math import m1p


def _t_k_q(k, q, i, chi, theta, gamma):
    """
    complex T{K, Q}(i, Omega)

    k = K: 0, 1, 2
    q = Q: -K ... K
    i: 0 ... 3 (denotes the Stokes component)

    Reference: Table 5.6
    See also: (5.159), (5.160)

    Input integrity implied.
    """
    if q < 0:
        return m1p(q) * t_k_q(k, -q, i, chi, theta, gamma).conjugate()

    if i == 0:
        if k == 0:
            return 1 + 0j
        if k == 1:
            return 0 + 0j
        if q == 0:
            return 0.5 / sqrt2 * (3 * cos(theta) ** 2 - 1) + 0j
        if q == 1:
            return -0.5 * sqrt3 * sin(theta) * cos(theta) * exp(1j * chi)
        return 0.25 * sqrt3 * sin(theta) ** 2 * exp(2j * chi)
    if i == 1:
        if k <= 1:
            return 0 + 0j
        if q == 0:
            return -1.5 / sqrt2 * cos(2 * gamma) * sin(theta) ** 2 + 0j
        if q == 1:
            return (
                -0.5
                * sqrt3
                * (cos(2 * gamma) * cos(theta) + 1j * sin(2 * gamma))
                * sin(theta)
                * exp(1j * chi)
            )
        return (
            -0.25
            * sqrt3
            * (
                cos(2 * gamma) * (1 + cos(theta) ** 2)
                + 2j * sin(2 * gamma) * cos(theta)
            )
            * exp(2j * chi)
        )
    if i == 2:
        if k <= 1:
            return 0 + 0j
        if q == 0:
            return 1.5 / sqrt2 * sin(2 * gamma) * sin(theta) ** 2 + 0j
        if q == 1:
            return (
                0.5
                * sqrt3
                * (sin(2 * gamma) * cos(theta) - 1j * cos(2 * gamma))
                * sin(theta)
                * exp(1j * chi)
            )
        return (
            0.25
            * sqrt3
            * (
                sin(2 * gamma) * (1 + cos(theta) ** 2)
                - 2j * cos(2 * gamma) * cos(theta)
            )
            * exp(2j * chi)
        )
    if k == 0 or k == 2:
        return 0 + 0j
    if q == 0:
        return sqrt3 / sqrt2 * cos(theta) + 0j
    return -0.5 * sqrt3 * sin(theta) * exp(1j * chi)


# TODO: pre-calculating first can improve performance
t_k_q = np.vectorize(_t_k_q)
