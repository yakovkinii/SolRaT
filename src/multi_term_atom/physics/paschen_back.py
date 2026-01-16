import logging
from functools import lru_cache
from typing import Tuple, Union

import numpy as np
from numpy import sqrt

from src.engine.functions.general import half_int_to_str
from src.engine.functions.looping import fromto
from src.common.constants import c_cm_sm1, h_erg_s, mu0_erg_gaussm1
from src.multi_term_atom.terms_levels_transitions.level_registry import Term


class PaschenBackEigenvalues:
    def __init__(self):
        self.data = dict()

    def set(self, j, M, value, term=None):
        if term is not None:
            logging.warning(
                "PaschenBackEigenvalues.set: term argument is deprecated and will be removed in future versions."
            )
        self.data[(half_int_to_str(j), half_int_to_str(M))] = value

    def __call__(self, j, M, term=None):
        if term is not None:
            logging.warning(
                "PaschenBackEigenvalues.set: term argument is deprecated and will be removed in future versions."
            )
        assert (half_int_to_str(j), half_int_to_str(M)) in self.data, (
            f"PaschenBackEigenvalues: {half_int_to_str(j)}, {half_int_to_str(M)}"
            f" not found. Please explicitly enforce triangular rules."
        )
        return self.data[(half_int_to_str(j), half_int_to_str(M))]


class PaschenBackCoefficients:
    def __init__(self):
        self.data = dict()

    def set(self, j, J, M, value, term=None):
        if term is not None:
            logging.warning(
                "PaschenBackCoefficients.set: term argument is deprecated and will be removed in future versions."
            )
        self.data[(half_int_to_str(j), half_int_to_str(J), half_int_to_str(M))] = value

    def __call__(self, j, J, M, term=None):
        if term is not None:
            logging.warning(
                "PaschenBackCoefficients.set: term argument is deprecated and will be removed in future versions."
            )
        return self.data[(half_int_to_str(j), half_int_to_str(J), half_int_to_str(M))]


def _g_ls(L, S, J, artificial_S_scale: Union[float, None] = None):
    """
    Reference: (3.8)
    """
    if J == 0:
        return 1
    if artificial_S_scale is not None:
        return 1 + 0.5 * artificial_S_scale * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / J / (J + 1)
    return 1 + 0.5 * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / J / (J + 1)


def get_artificial_S_scale_from_term_g(g, J, L, S):
    """
    This back-derives scale from _g_ls() given g value.
    """
    alpha = 0.5 * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / J / (J + 1)
    return (g-1) / alpha


@lru_cache(maxsize=None)  # Todo check if this works correctly
def calculate_paschen_back(
    term: Term, magnetic_field_gauss: float
) -> Tuple[PaschenBackEigenvalues, PaschenBackCoefficients]:
    """
    Reference: (3.61 a b)
    """
    eigenvalues = PaschenBackEigenvalues()
    coefficients = PaschenBackCoefficients()

    L = term.L
    S = term.S

    J_max = L + S
    J_min = abs(L - S)
    for M in fromto(-J_max, J_max):
        # For each fixed M (which is eigenvalue of Jz),
        # we can couple only J >= |M|.
        # Also, J_min <= J <= J_max
        # Therefore coupled J are [max(J_min, |M|) ... J_max]
        # Matrix block size is therefore J_max - max(J_min, |M|) + 1

        min_J_for_M = max(J_min, abs(M))
        block_size = int(J_max - min_J_for_M + 1)

        # M = const
        #
        # i=0   i=1     i=2
        # J_max J_max-1 J_max-2 ...
        # V     V       X       J_max   i=0
        # V     V       V       J_max-1 i=1
        # X     V       V       J_max-2 i=2
        #                       ...
        #
        # We have 3-diagonal matrix block_size x block_size
        matrix = np.zeros((block_size, block_size))

        mu0b_cm = mu0_erg_gaussm1 * magnetic_field_gauss / h_erg_s / c_cm_sm1  # mu_0 * B in cm-1
        for i in range(block_size):
            J = J_max - i  # J of current row

            # Fill diagonal elements
            if term.artificial_S_scale is not None:
                matrix[i, i] = (
                    term.get_level(J).energy_cmm1
                    + mu0b_cm * _g_ls(L, S, J, artificial_S_scale=term.artificial_S_scale) * M
                )
            else:
                matrix[i, i] = term.get_level(J).energy_cmm1 + mu0b_cm * _g_ls(L, S, J) * M

            # Fill non-diagonal elements
            if i + 1 < block_size:  # if i+1 is still a valid index, fill <J-1| H_B |J>
                value = (-mu0b_cm / 2 / J) * sqrt(
                    (J + S + L + 1)
                    * (J - S + L)
                    * (J + S - L)
                    * (-J + S + L + 1)
                    * (J**2 - M**2)
                    / (2 * J + 1)
                    / (2 * J - 1)
                )
                if term.artificial_S_scale is not None:
                    value = value * term.artificial_S_scale

                matrix[i, i + 1] = value
                matrix[i + 1, i] = value
        eig_values, eig_vectors = np.linalg.eig(matrix)

        # eigenvectors is a matrix where columns are eigenvectors => column number is j_small
        # row number is index of j; j = j_max - row_number
        for j in range(block_size):
            eigenvalues.set(j=J_max - j, M=M, value=eig_values[j])
            for j1 in range(block_size):
                coefficients.set(j=J_max - j, M=M, J=J_max - j1, value=eig_vectors[j1, j])

    return eigenvalues, coefficients
