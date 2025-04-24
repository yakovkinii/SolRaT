import logging
from typing import Tuple

import numpy as np
from numpy import sqrt

from core.base.python import half_int_to_str, range_inclusive
from core.terms_levels_transitions.term_registry import Level
from core.utility.constant import c, h, mu0


class PaschenBackEigenvalues:
    def __init__(self):
        self.data = dict()

    def set(self, j, level: Level, M, value):
        self.data[(half_int_to_str(j), level.level_id, half_int_to_str(M))] = value

    def __call__(self, j, level: Level, M):
        if (half_int_to_str(j), level.level_id, half_int_to_str(M)) not in self.data:
            logging.warning(
                f"PaschenBackEigenvalues: {half_int_to_str(j)}, {level.level_id}, {half_int_to_str(M)} not found."
            )
            return 0
        return self.data[(half_int_to_str(j), level.level_id, half_int_to_str(M))]


class PaschenBackCoefficients:
    def __init__(self):
        self.data = dict()

    def set(self, j, J, level: Level, M, value):
        self.data[(half_int_to_str(j), half_int_to_str(J), level.level_id, half_int_to_str(M))] = value

    def __call__(self, j, J, level: Level, M):
        if (half_int_to_str(j), half_int_to_str(J), level.level_id, half_int_to_str(M)) not in self.data:
            logging.warning(
                f"PaschenBackCoefficients: {half_int_to_str(j)}, {half_int_to_str(J)}, {level.level_id}, {half_int_to_str(M)} not found."
            )
            logging.info(
                self.data.keys()
            )
            return 0
        return self.data[(half_int_to_str(j), half_int_to_str(J), level.level_id, half_int_to_str(M))]


def calculate_paschen_back(
    level: Level, magnetic_field_gauss: float
) -> Tuple[PaschenBackEigenvalues, PaschenBackCoefficients]:
    """
    Reference: (3.61 a b)
    """
    eigenvalues = PaschenBackEigenvalues()
    coefficients = PaschenBackCoefficients()

    L = level.l
    S = level.s

    def _g_ls(l, s, j):
        """
        Reference: (3.8)
        """
        if j == 0:
            return 1

        return 1 + 0.5 * (j * (j + 1) + s * (s + 1) - l * (l + 1)) / j / (j + 1)

    J_max = L + S
    J_min = abs(L - S)
    for M in range_inclusive(-J_max, J_max):
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

        mu0b_cm = mu0 * magnetic_field_gauss / h / c  # mu_0 * B in cm-1
        for i in range(block_size):
            J = J_max - i  # J of current row

            # Fill diagonal elements
            matrix[i, i] = level.get_term(J).energy_cmm1 + mu0b_cm * _g_ls(L, S, J) * M

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
                matrix[i, i + 1] = value
                matrix[i + 1, i] = value
        eig_values, eig_vectors = np.linalg.eig(matrix)

        # eigenvectors is a matrix where columns are eigenvectors => column number is j_small
        # row number is index of j; j = j_max - row_number
        for j in range(block_size):
            eigenvalues.set(j=J_max-j, level=level, M=M, value=eig_values[j])
            for j1 in range(block_size):
                coefficients.set(j=J_max-j, level=level, M=M, J=J_max-j1, value=eig_vectors[j1, j])

    return eigenvalues, coefficients
