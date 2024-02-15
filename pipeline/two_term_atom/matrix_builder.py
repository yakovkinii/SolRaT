from typing import List

import numpy as np

from core.utility.python import (
    half_int_to_str,
    triangular,
    projection,
)
from pipeline.two_term_atom.term_registry import Level


class MatrixBuilder:
    def __init__(self, levels: List[Level]):
        """
        This class helps to build the matrix for rhos.
        All possible rhos are defined by terms.
        """
        # Create mapping [level_id, K, Q, J, J'] <-> matrix index
        self.index_to_parameters = dict()
        self.coherence_id_to_index = dict()

        index = 0
        for level in levels:
            for j in triangular(level.l, level.s):
                for j_prime in triangular(level.l, level.s):
                    for k in triangular(j, j_prime):
                        for q in projection(k):
                            coherence_id = self.construct_coherence_id(
                                level=level, k=k, q=q, j=j, j_prime=j_prime
                            )
                            self.coherence_id_to_index[coherence_id] = index
                            self.index_to_parameters[index] = (
                                level.level_id,
                                k,
                                q,
                                j,
                                j_prime,
                            )
                            index += 1
        # create the matrix
        matrix_size = index
        self.rho_matrix = np.zeros((matrix_size, matrix_size), dtype=np.complex128)
        self.selected_coherence = None

    @staticmethod
    def construct_coherence_id(level: Level, k: int, q: int, j: float, j_prime: float):
        return f"{level.level_id}_k{k}_q{q}_j{half_int_to_str(j)}_jp{half_int_to_str(j_prime)}"

    def reset_matrix(self):
        self.rho_matrix = self.rho_matrix * 0

    def select_equation(
        self,
        level: Level,
        k: int,
        q: int,
        j: float,
        j_prime: float,
    ):
        """
        Selects the equation to add coefficients to.
        """
        coherence_id = self.construct_coherence_id(
            level=level, k=k, q=q, j=j, j_prime=j_prime
        )
        self.selected_coherence = coherence_id

    def add_coefficient(
        self,
        level: Level,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        coefficient: float,
    ):
        """
        Adds a coefficient to the selected equation.
        """
        if coefficient == 0:
            return

        index0 = self.coherence_id_to_index[self.selected_coherence]
        coherence_id = self.construct_coherence_id(
            level=level, k=k, q=q, j=j, j_prime=j_prime
        )
        assert coherence_id in self.coherence_id_to_index.keys(), (
            f"Trying to add coefficient to non-existing " f"coherence {coherence_id}"
        )
        index1 = self.coherence_id_to_index[coherence_id]
        print(f"=== {index0} {index1} += {coefficient}")
        self.rho_matrix[index0, index1] += coefficient
