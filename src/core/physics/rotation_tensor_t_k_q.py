from functools import lru_cache

from numpy import cos, exp, sin

from src.core.engine.functions.general import m1p
from src.core.physics.constants import sqrt2, sqrt3


@lru_cache(maxsize=None)
def t_k_q(K, Q, stokes_component_index, chi, theta, gamma):
    """
    complex T{K, Q}(i, Omega)

    k = K: 0, 1, 2
    q = Q: -K ... K
    i: 0 ... 3 (denotes the Stokes component)

    Reference: Table 5.6
    See also: (5.159), (5.160)

    Input integrity implied.
    """
    if Q < 0:
        return m1p(Q) * t_k_q(K, -Q, stokes_component_index, chi, theta, gamma).conjugate()

    if stokes_component_index == 0:
        if K == 0:
            return 1 + 0j
        if K == 1:
            return 0 + 0j
        if Q == 0:
            return 0.5 / sqrt2 * (3 * cos(theta) ** 2 - 1) + 0j
        if Q == 1:
            return -0.5 * sqrt3 * sin(theta) * cos(theta) * exp(1j * chi)
        return 0.25 * sqrt3 * sin(theta) ** 2 * exp(2j * chi)
    if stokes_component_index == 1:
        if K <= 1:
            return 0 + 0j
        if Q == 0:
            return -1.5 / sqrt2 * cos(2 * gamma) * sin(theta) ** 2 + 0j
        if Q == 1:
            return -0.5 * sqrt3 * (cos(2 * gamma) * cos(theta) + 1j * sin(2 * gamma)) * sin(theta) * exp(1j * chi)
        return (
            -0.25 * sqrt3 * (cos(2 * gamma) * (1 + cos(theta) ** 2) + 2j * sin(2 * gamma) * cos(theta)) * exp(2j * chi)
        )
    if stokes_component_index == 2:
        if K <= 1:
            return 0 + 0j
        if Q == 0:
            return 1.5 / sqrt2 * sin(2 * gamma) * sin(theta) ** 2 + 0j
        if Q == 1:
            return 0.5 * sqrt3 * (sin(2 * gamma) * cos(theta) - 1j * cos(2 * gamma)) * sin(theta) * exp(1j * chi)
        return (
            0.25 * sqrt3 * (sin(2 * gamma) * (1 + cos(theta) ** 2) - 2j * cos(2 * gamma) * cos(theta)) * exp(2j * chi)
        )
    if K == 0 or K == 2:
        return 0 + 0j
    if Q == 0:
        return sqrt3 / sqrt2 * cos(theta) + 0j
    return -0.5 * sqrt3 * sin(theta) * exp(1j * chi)
