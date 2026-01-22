import logging
import unittest

import numpy as np
from yatools import logging_config

from src.common.constants import sqrt2
from src.common.rotations import WignerD
from src.engine.functions.looping import FROMTO, PROJECTION
from src.engine.generators.nested_loops import nested_loops


def d_1_M_N(M, N, beta):
    """
    Table 2.1 (p. 57)
    """
    C = np.cos(beta)
    S = np.sin(beta)
    if M == -1:
        if N == -1:
            return 0.5 * (1 + C)
        if N == 0:
            return S / sqrt2
        return 0.5 * (1 - C)
    if M == 0:
        if N == -1:
            return -S / sqrt2
        if N == 0:
            return C
        return S / sqrt2
    if M == 1:
        if N == -1:
            return 0.5 * (1 - C)
        if N == 0:
            return -S / sqrt2
        return 0.5 * (1 + C)


def d_2_M_N(M, N, beta):
    """
    Table 2.1 (p. 57)
    """
    C = np.cos(beta)
    S = np.sin(beta)

    if M == -2:
        if N == -2:
            return 0.25 * (1 + C) ** 2
        if N == -1:
            return 0.5 * S * (1 + C)
        if N == 0:
            return np.sqrt(3 / 8) * S**2
        if N == 1:
            return 0.5 * S * (1 - C)
        return 0.25 * (1 - C) ** 2

    if M == -1:
        if N == -2:
            return -0.5 * S * (1 + C)
        if N == -1:
            return (C - 0.5) * (1 + C)
        if N == 0:
            return np.sqrt(3 / 2) * S * C
        if N == 1:
            return (C + 0.5) * (1 - C)
        return 0.5 * S * (1 - C)

    if M == 0:
        if N == -2:
            return np.sqrt(3 / 8) * S**2
        if N == -1:
            return -np.sqrt(3 / 2) * S * C
        if N == 0:
            return 0.5 * (3 * C**2 - 1)
        if N == 1:
            return np.sqrt(3 / 2) * S * C
        return np.sqrt(3 / 8) * S**2

    if M == 1:
        if N == -2:
            return -0.5 * S * (1 - C)
        if N == -1:
            return (C + 0.5) * (1 - C)
        if N == 0:
            return -np.sqrt(3 / 2) * S * C
        if N == 1:
            return (C - 0.5) * (1 + C)
        return 0.5 * S * (1 + C)

    if M == 2:
        if N == -2:
            return 0.25 * (1 - C) ** 2
        if N == -1:
            return -0.5 * S * (1 - C)
        if N == 0:
            return np.sqrt(3 / 8) * S**2
        if N == 1:
            return -0.5 * S * (1 + C)
        return 0.25 * (1 + C) ** 2


def D_1_M_N(M, N, alpha, beta, gamma):
    return d_1_M_N(M, N, beta) * np.exp(-1j * (M * alpha + N * gamma))


def D_2_M_N(M, N, alpha, beta, gamma):
    return d_2_M_N(M, N, beta) * np.exp(-1j * (M * alpha + N * gamma))


class TestWignerD(unittest.TestCase):
    def test_wigner_D(self):
        """
        Compare T{K, Q} calculated from table 5.5 with T{K, Q} calculated from Wigner D function.
        This simultaneously tests t{K, P}, T{K, Q}, and Wigner D function.
        """
        logging_config.init(logging.INFO)

        chi = np.pi / 5
        theta = np.pi / 3
        gamma = np.pi / 7

        D = WignerD(alpha=chi, beta=theta, gamma=gamma, K_max=2)

        for K, M, N in nested_loops(
            K=FROMTO(1, 2),
            M=PROJECTION("K"),
            N=PROJECTION("K"),
        ):
            D1 = D_1_M_N(M, N, chi, theta, gamma) if K == 1 else D_2_M_N(M, N, chi, theta, gamma)
            D2 = D(K=K, P=M, Q=N)
            assert np.max(np.abs(D1 - D2)) < 1e-12
