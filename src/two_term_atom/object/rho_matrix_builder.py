from typing import List

import numpy as np
from numpy import sqrt

from src.core.engine.functions.general import half_int_to_str
from src.core.engine.functions.looping import PROJECTION, TRIANGULAR
from src.core.engine.generators.nested_loops import nested_loops
from src.two_term_atom.terms_levels_transitions.term_registry import Level


def _construct_coherence_id(level: Level, K: float, Q: float, J: float, Jʹ: float):
    return _construct_coherence_id_from_level_id(level_id=level.level_id, K=K, Q=Q, J=J, Jʹ=Jʹ)


def _construct_coherence_id_from_level_id(level_id: str, K: float, Q: float, J: float, Jʹ: float):
    return f"{level_id}_K={half_int_to_str(K)}_Q={half_int_to_str(Q)}_J={half_int_to_str(J)}_Jʹ={half_int_to_str(Jʹ)}"


class Rho:
    def __init__(self):
        self.data = dict()

    def set_from_level_id(self, level_id: str, K: float, Q: float, J: float, Jʹ: float, value: np.ndarray):
        self.data[_construct_coherence_id_from_level_id(level_id=level_id, K=K, Q=Q, J=J, Jʹ=Jʹ)] = value

    def __call__(self, level: Level, K: float, Q: float, J: float, Jʹ: float):
        coherence_id = _construct_coherence_id(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
        return self.data[coherence_id]


class RhoMatrixBuilder:
    def __init__(self, levels: List[Level], n_frequencies: int):
        """
        This class helps to build the matrix for rhos.
        All possible rhos are defined by terms.
        """

        # Create mapping [level_id, K, Q, J, Jʹ] <-> matrix index
        self.index_to_parameters = dict()
        self.coherence_id_to_index = dict()
        self.trace_indexes = []
        self.trace_weights = []
        index = 0
        for level in levels:
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(level.L, level.S),
                Jʹ=TRIANGULAR(level.L, level.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                coherence_id = _construct_coherence_id(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.coherence_id_to_index[coherence_id] = index
                self.index_to_parameters[index] = (level.level_id, K, Q, J, Jʹ)
                if K == 0 and Q == 0 and J == Jʹ:
                    self.trace_indexes.append(index)
                    self.trace_weights.append(sqrt(2 * J + 1))
                index += 1

        # create the matrix
        matrix_size = index
        self.rho_matrix = np.zeros((n_frequencies, matrix_size, matrix_size), dtype=np.complex128)
        self.selected_coherence = None

    def reset_matrix(self):
        self.rho_matrix = self.rho_matrix * 0

    def select_equation(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        """
        Selects the equation to add coefficients to.
        """
        coherence_id = _construct_coherence_id(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
        self.selected_coherence = coherence_id

    def add_coefficient(self, level: Level, K: int, Q: int, J: float, Jʹ: float, coefficient: np.array):
        """
        Adds a coefficient to the selected equation.
        """

        if not isinstance(coefficient, np.ndarray):
            coefficient = np.array([coefficient] * self.rho_matrix.shape[0], dtype=np.complex128)

        assert isinstance(coefficient, np.ndarray), "Coefficient must be a numpy array"
        assert coefficient.ndim == 1, "Coefficient must be a 1D array"
        assert coefficient.shape[0] == self.rho_matrix.shape[0]

        if coefficient.__eq__(0).all():
            return

        index0 = self.coherence_id_to_index[self.selected_coherence]
        coherence_id = _construct_coherence_id(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
        assert coherence_id in self.coherence_id_to_index.keys(), (
            f"Trying to add coefficient to non-existing " f"coherence {coherence_id}"
        )
        index1 = self.coherence_id_to_index[coherence_id]
        self.rho_matrix[:, index0, index1] += coefficient
