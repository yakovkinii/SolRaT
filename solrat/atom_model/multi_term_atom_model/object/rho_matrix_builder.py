from typing import List

import numpy as np
import pandas as pd
from numpy import sqrt

from solrat.atom_model.base_atom_model.object.rho import BaseRho
from solrat.atom_model.multi_term_atom_model.object.level_registry import Term
from solrat.engine.functions.decorators import log_method
from solrat.engine.functions.general import half_int_to_str
from solrat.engine.functions.looping import PROJECTION, TRIANGULAR
from solrat.engine.generators.nested_loops import nested_loops


def construct_coherence_id(term: Term, K: float, Q: float, J: float, Jʹ: float):
    """
    Construct a unique ID for a coherence
    """
    return construct_coherence_id_from_term_id(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)


def construct_coherence_id_from_term_id(term_id: str, K: float, Q: float, J: float, Jʹ: float):
    """
    Construct a unique ID for a coherence
    """
    return f"{term_id}_K={half_int_to_str(K)}_Q={half_int_to_str(Q)}_J={half_int_to_str(J)}_Jʹ={half_int_to_str(Jʹ)}"


class Rho(BaseRho):
    """
    A container for the density tensor Rho
    :param terms: list of all terms.
    """

    def __init__(self, terms: List[Term]):
        self.data = dict()
        self.terms = {term.term_id: term for term in terms}
        self.datarows = []
        # Cached dataframe view of datarows, built lazily on first get_vector() call.
        self._datarows_df = None

    def set_from_term_id(self, term_id: str, K: float, Q: float, J: float, Jʹ: float, value: complex):
        """
        Set the value
        """
        self.data[construct_coherence_id_from_term_id(term_id=term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)] = value
        self.datarows.append({"term_id": term_id, "K": K, "Q": Q, "J": J, "Jʹ": Jʹ, "value": value})

    def get_vector(self, term_id, K, Q, J, Jʹ):
        """
        Vectorized get: look up many coherences at once (testing path).

        The engine passes each argument as ``[N, 1]``; flatten to 1-D, merge against the
        stored rows, and return a 1-D ``[N]`` array of values. Falls back to building the
        dataframe from ``datarows`` if the Rho was populated the old (scalar) way.
        """
        if self._datarows_df is None:
            self._datarows_df = pd.DataFrame(self.datarows)
        query = pd.DataFrame(
            {
                "term_id": np.asarray(term_id).reshape(-1),
                "K": np.asarray(K, dtype=float).reshape(-1),
                "Q": np.asarray(Q, dtype=float).reshape(-1),
                "J": np.asarray(J, dtype=float).reshape(-1),
                "Jʹ": np.asarray(Jʹ, dtype=float).reshape(-1),
            }
        )
        merged = query.merge(self._datarows_df, on=["term_id", "K", "Q", "J", "Jʹ"], how="left")
        return merged["value"].to_numpy()

    def __call__(self, K: float, Q: float, J: float, Jʹ: float, term_id: str) -> np.complex128:
        """
        Get the value (scalar, old path).
        """
        coherence_id = construct_coherence_id_from_term_id(term_id=term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
        return self.data[coherence_id]


class RhoMatrixBuilder:
    """
    This class helps to build the matrix for rhos.
    All possible rhos are defined by terms.

    :param terms: list of all terms.
    """

    def __init__(self, terms: List[Term]):
        # Create mapping [term_id, K, Q, J, Jʹ] <-> matrix index
        self.index_to_parameters = dict()
        self.coherence_id_to_index = dict()
        self.trace_indexes = []
        self.trace_weights = []
        # Parameter arrays in index order (used to build the index lookup table below).
        param_term_ids, param_K, param_Q, param_J, param_Jʹ = [], [], [], [], []
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
                param_term_ids.append(term.term_id)
                param_K.append(K)
                param_Q.append(Q)
                param_J.append(J)
                param_Jʹ.append(Jʹ)
                if K == 0 and Q == 0 and J == Jʹ:
                    self.trace_indexes.append(index)
                    self.trace_weights.append(sqrt(2 * J + 1))
                index += 1
        self.param_term_ids = np.array(param_term_ids, dtype=object)
        self.param_K = np.array(param_K, dtype=float)
        self.param_Q = np.array(param_Q, dtype=float)
        self.param_J = np.array(param_J, dtype=float)
        self.param_Jʹ = np.array(param_Jʹ, dtype=float)
        # Lookup table (term_id, K, Q, J, Jʹ) -> matrix index, for vectorized index assignment.
        self.param_index_df = pd.DataFrame(
            {
                "term_id": self.param_term_ids,
                "K": self.param_K,
                "Q": self.param_Q,
                "J": self.param_J,
                "Jʹ": self.param_Jʹ,
                "index": np.arange(len(self.param_K)),
            }
        )

        # create the matrix
        matrix_size = index
        self.rho_matrix = np.zeros((matrix_size, matrix_size), dtype=np.complex128)
        self.selected_coherence = None

    @log_method
    def reset_matrix(self):
        """
        Reset the matrix to fill it from scratch later.
        """
        self.rho_matrix = self.rho_matrix * 0

    def select_equation(self, term: Term, K: int, Q: int, J: float, Jʹ: float):
        r"""
        Selects the equation to add coefficients to.

        The equation is of a form :math:`M_{ij} \rho_j = 0`. Here we select i.
        """
        coherence_id = construct_coherence_id(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ)
        self.selected_coherence = coherence_id

    def add_coefficient(self, term: Term, K: int, Q: int, J: float, Jʹ: float, coefficient: complex):
        r"""
        Adds a coefficient to the selected equation.

        The equation is of a form :math:`M_{ij} \rho_j = 0`. We have already selected i.
        Now we do :math:`M_{ij} += coefficient`, where j is defined by {term, :math:`K, Q, J, Jʹ`}.
        """
        coefficient = complex(coefficient)
        assert isinstance(coefficient, complex), "Coefficient must be complex"
        coherence_id = construct_coherence_id(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ)

        if coefficient == 0:
            return

        index0 = self.coherence_id_to_index[self.selected_coherence]
        if coherence_id not in self.coherence_id_to_index.keys():
            raise ValueError(f"Trying to add coefficient to non-existing coherence {coherence_id}")  # pragma: no cover
        index1 = self.coherence_id_to_index[coherence_id]
        self.rho_matrix[index0, index1] += coefficient

    @log_method
    def add_coefficient_from_df(self, df: pd.DataFrame):
        r"""
        Adds a coefficient to the selected equation from the dataframe of the form "index0", "index1", "coefficient".

        The equation is of a :math:`M_{ij} \rho_j = 0`.
        For each df row, we do :math:`M_{ij} += coefficient`, where i=index0 and j=index1.
        """
        df = df[["index0", "index1", "coefficient"]].groupby(["index0", "index1"]).sum().reset_index()
        self.rho_matrix[df.index0, df.index1] += df.coefficient
