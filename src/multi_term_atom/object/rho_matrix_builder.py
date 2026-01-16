import logging
from typing import List, Union

import numpy as np
from numpy import sqrt

from src.engine.functions.general import half_int_to_str
from src.engine.functions.looping import PROJECTION, TRIANGULAR
from src.engine.generators.nested_loops import nested_loops
from src.multi_term_atom.terms_levels_transitions.level_registry import Term

construct_coherence_id_warned = False


def construct_coherence_id(term: Term, K: float, Q: float, J: float, Jʹ: float, term_id:str=None):
    global construct_coherence_id_warned
    if term is not None:
        term_id = term.term_id
        if not construct_coherence_id_warned:
            logging.warning("construct_coherence_id: term_id argument is deprecated and will be removed in future versions.")
            construct_coherence_id_warned = True
    assert term_id is not None
    return construct_coherence_id_from_term_id(term_id=term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)


def construct_coherence_id_from_term_id(term_id: str, K: float, Q: float, J: float, Jʹ: float):
    return f"{term_id}_K={half_int_to_str(K)}_Q={half_int_to_str(Q)}_J={half_int_to_str(J)}_Jʹ={half_int_to_str(Jʹ)}"


class Rho:
    def __init__(self, terms: List[Term]):
        self.data = dict()
        self.terms = terms

    def set_from_term_id(self, term_id: str, K: float, Q: float, J: float, Jʹ: float, value: np.ndarray):
        self.data[construct_coherence_id_from_term_id(term_id=term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)] = value

    def __call__(self, K: float, Q: float, J: float, Jʹ: float,  term: Union[Term, str]=None, term_id=None):
        coherence_id = construct_coherence_id(term=term, term_id=term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
        return self.data[coherence_id]


class RhoMatrixBuilder:
    def __init__(self, terms: List[Term], n_frequencies: int):
        """
        This class helps to build the matrix for rhos.
        All possible rhos are defined by levels.
        """
        assert n_frequencies == 1, "RhoMatrixBuilder only supports n_frequencies == 1"

        # Create mapping [term_id, K, Q, J, Jʹ] <-> matrix index
        self.index_to_parameters = dict()
        self.coherence_id_to_index = dict()
        self.trace_indexes = []
        self.trace_weights = []
        index = 0
        for term in terms:
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                coherence_id = construct_coherence_id(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.coherence_id_to_index[coherence_id] = index
                self.index_to_parameters[index] = (term.term_id, K, Q, J, Jʹ)
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

    def select_equation(self, term: Term, K: int, Q: int, J: float, Jʹ: float):
        """
        Selects the equation to add coefficients to.
        """
        coherence_id = construct_coherence_id(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ)
        self.selected_coherence = coherence_id

    def add_coefficient(self, term: Term, K: int, Q: int, J: float, Jʹ: float, coefficient: np.array):
        """
        Adds a coefficient to the selected equation.
        """

        if not isinstance(coefficient, np.ndarray):
            coefficient = np.array([coefficient] * self.rho_matrix.shape[0], dtype=np.complex128)

        assert isinstance(coefficient, np.ndarray), "Coefficient must be a numpy array"
        assert coefficient.ndim == 1, "Coefficient must be a 1D array"
        assert coefficient.shape[0] == self.rho_matrix.shape[0]
        coherence_id = construct_coherence_id(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ)

        if coefficient.__eq__(0).all():
            # logging.warning("Coefficient is zero, skipping addition.")
            # logging.info(f'selected_coherence: {self.selected_coherence}')
            # logging.info(f'coherence_id: {coherence_id}')
            return

        index0 = self.coherence_id_to_index[self.selected_coherence]
        assert coherence_id in self.coherence_id_to_index.keys(), (
            f"Trying to add coefficient to non-existing " f"coherence {coherence_id}"
        )
        index1 = self.coherence_id_to_index[coherence_id]
        self.rho_matrix[:, index0, index1] += coefficient

    def add_coefficient_from_df(self, df):
        df = df[["index0", "index1", "coefficient"]].groupby(["index0", "index1"]).sum().reset_index()
        self.rho_matrix[0, df.index0, df.index1] += df.coefficient
