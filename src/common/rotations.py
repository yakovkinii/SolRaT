"""
TODO
TODO  This file needs improved documentation.
TODO
"""

from functools import lru_cache

import numpy as np
import sympy
from numpy import cos, exp, sin
from sympy.physics.wigner import wigner_d

from src.engine.functions.general import delta, m1p
from src.engine.functions.looping import PROJECTION, fromto
from src.engine.generators.nested_loops import nested_loops
from src.common.constants import sqrt2, sqrt3


class WignerD:
    """
    Wigner D function.
    alpha, beta, gamma are Euler angles in radians.
    Typically, we have alpha = chi, beta = theta, gamma = gamma (see Fig. 5.14).
    """

    def __init__(self, alpha, beta, gamma, K_max):
        self.d = {}
        for K in fromto(0, K_max):
            # Note: sympy uses a different convention for angles, so I perform under-the-hood conversion here.
            self.d[K] = wigner_d(J=int(K), alpha=-alpha, beta=-beta, gamma=-gamma)

    @lru_cache(maxsize=None)
    def __call__(self, K, P, Q):
        result = np.array(sympy.N(self.d[K][int(K - P), int(K - Q)])).astype(np.complex128)
        return result


def t_K_P(K, P, stokes_component_index):
    """
    t{K, P}(i)
    Reference: Table 5.5
    This is implemented primarily to validate Wigner D functions
    """
    if K == 0:
        return delta(P, 0) * delta(stokes_component_index, 0)
    if K == 1:
        return sqrt3 / sqrt2 * delta(P, 0) * delta(stokes_component_index, 3)
    return (
        1 / sqrt2 * delta(P, 0) * delta(stokes_component_index, 0)
        - sqrt3 / 2 * (delta(P, -2) + delta(P, 2)) * delta(stokes_component_index, 1)
        + 1j * sqrt3 / 2 * (delta(P, -2) - delta(P, 2)) * delta(stokes_component_index, 2)
    )


# @lru_cache(maxsize=None)
def T_K_Q_double_rotation(K, Q, stokes_component_index, D_inverse_omega: WignerD, D_magnetic: WignerD):
    """
    (5.159), (2.74), (5.122)
    Two consecutive D rotations within T tensor.
    """
    result = 0
    for P, Qʹ in nested_loops(P=PROJECTION(K), Qʹ=PROJECTION(K)):
        result = result + t_K_P(
            K=K,
            P=P,
            stokes_component_index=stokes_component_index,
        ) * D_inverse_omega(
            K=K, P=P, Q=Qʹ
        ) * D_magnetic(K=K, P=Qʹ, Q=Q)

    return result


@lru_cache(maxsize=None)
def T_K_Q(K, Q, stokes_component_index, chi, theta, gamma):
    """
    T{K, Q}(i, Omega)

    Reference: Table 5.6
    See also: (5.159), (5.160)
    """
    if Q < 0:
        return m1p(Q) * T_K_Q(K, -Q, stokes_component_index, chi, theta, gamma).conjugate()

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
