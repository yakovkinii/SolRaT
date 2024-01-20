import numpy as np
from numpy import sqrt

from core.utility.constant import mu0, h, c
from core.utility.python import range_inclusive


def g_ls(l, s, j):
    """
    float g(J, L, S)
    Reference: (3.8)
    """
    return 1 + 0.5 * (j * (j + 1) + s * (s + 1) - l * (l + 1)) / j / (j + 1)


def e_dict_lower_d3():
    # 2p^3P = there is one electron in 2p orbital (the other one in 1s is omitted); 2S+1 = 3 (=> S=1); L=1;
    # L = 1
    # S = 1
    # J = 0, 1, 2

    # It returns the energy at different J (just take from NIST) in cm-1. Can include/exclude H0,
    # should be taken into account in RTE
    return {0: 169087.8291, 1: 169086.8412, 2: 169086.7647}


def e_dict_lower_Ha():
    return {
        0.5: 82258.9191133,
        1.5: 82259.2850014,
    }


def paschen_back_diagonalization(l, s, b, e: dict):
    """
    float C{j_small, J}(..., M)
    float lambda{j_small}(..., M)

    encoded as
    c[m][j_small][l]
    lambda[m][j_small]

    Reference: (3.61 a b)

    b is magnetic field in Gauss
    e[j] is a dict of energies for different J in cm-1.
    e[j] can be found in NIST tables.
    """
    all_coefficients = dict()
    all_lambdas = dict()
    j_max = l + s
    j_min = abs(l - s)
    for m in range_inclusive(-j_max, j_max):
        # For each fixed M (which is eigenvalue of Jz),
        # we can couple only J >= |M|.
        # Also, J_min <= J <= J_max
        # Therefore coupled J are [max(J_min, |M|) ... J_max]
        # Matrix block size is therefore J_max - max(J_min, |M|) + 1
        block_size = int(j_max - max(j_min, abs(m)) + 1)

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

        mu0b_cm = mu0 * b / h / c  # mu_0 * B in cm-1
        for i in range(block_size):
            j = j_max - i  # J of current row

            # Fill diagonal elements
            matrix[i, i] = e[j] + mu0b_cm * g_ls(l, s, j) * m

            # Fill non-diagonal elements
            if i + 1 < block_size:  # if i+1 is still a valid index, fill <J-1| H_B |J>
                value = (
                    -mu0b_cm
                    / 2
                    / j
                    * sqrt(
                        (j + s + l + 1)
                        * (j - s + l)
                        * (j + s - l)
                        * (-j + s + l + 1)
                        * (j**2 - m**2)
                        / (2 * j + 1)
                        / (2 * j - 1)
                    )
                )
                matrix[i, i + 1] = value
                matrix[i + 1, i] = value
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        c_coefficient = dict()
        lambda_coefficient = dict()
        # eigenvectors is a matrix where columns are eigenvectors => column number is j_small
        # row number is index of j; j = j_max - row_number
        for j_small in range(block_size):
            c_coefficient = dict()
            lambda_coefficient[j_small] = eigenvalues[j_small]
            for j in range(block_size):
                c_coefficient[j_small][j] = eigenvectors[j, j_small]

        all_lambdas[m] = lambda_coefficient
        all_coefficients[m] = c_coefficient

    return all_lambdas, all_coefficients
