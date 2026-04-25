from typing import List

import numpy as np
import pandas as pd
from numpy import sqrt

from solrat.atom_model.base_atom_model.object.rho import BaseRho
from solrat.atom_model.multi_level_atom_model.object.level_registry import Level
from solrat.engine.functions.decorators import log_method
from solrat.engine.functions.general import half_int_to_str
from solrat.engine.functions.looping import PROJECTION, TRIANGULAR
from solrat.engine.generators.nested_loops import nested_loops


def construct_coherence_id(level: Level, K: float, Q: float) -> str:
    """
    Construct a unique ID for a coherence in the Multi-Level model.
    """
    return construct_coherence_id_from_level_id(level_id=level.level_id, K=K, Q=Q)


def construct_coherence_id_from_level_id(level_id: str, K: float, Q: float) -> str:
    """
    Construct a unique ID for a coherence in the Multi-Level model.
    """
    return f"{level_id}_K={half_int_to_str(K)}_Q={half_int_to_str(Q)}"


class Rho(BaseRho):
    r"""
    Container for the Multi-Level density tensor :math:`\rho^K_Q(\alpha J)`.

    :param levels:  list of all levels.
    """

    def __init__(self, levels: List[Level]):
        self.data = dict()
        self.levels = {level.level_id: level for level in levels}

    def set_from_level_id(self, level_id: str, K: float, Q: float, value: complex):
        """
        Set the value of :math:`\rho^K_Q(\alpha J)`.
        """
        self.data[construct_coherence_id_from_level_id(level_id=level_id, K=K, Q=Q)] = value

    def __call__(self, K: float, Q: float, level_id: str) -> np.complex128:
        """
        Get the value of :math:`\rho^K_Q(\alpha J)`.
        """
        coherence_id = construct_coherence_id_from_level_id(level_id=level_id, K=K, Q=Q)
        return self.data[coherence_id]


class RhoMatrixBuilder:
    r"""
    Builder for the linear-system matrix that represents the Multi-Level SEE.

    Coherences are :math:`\rho^K_Q(\alpha J)` with :math:`K = 0 \ldots 2J` and
    :math:`Q = -K \ldots K`.

    :param levels:  list of all levels.
    """

    def __init__(self, levels: List[Level]):
        self.index_to_parameters = dict()
        self.coherence_id_to_index = dict()
        self.trace_indexes = []
        self.trace_weights = []
        index = 0
        for level in levels:
            for K, Q in nested_loops(
                K=TRIANGULAR(level.J, level.J),
                Q=PROJECTION("K"),
            ):
                coherence_id = construct_coherence_id(level=level, K=K, Q=Q)
                self.coherence_id_to_index[coherence_id] = index
                self.index_to_parameters[index] = (level.level_id, K, Q)
                if K == 0 and Q == 0:
                    self.trace_indexes.append(index)
                    self.trace_weights.append(sqrt(2 * level.J + 1))
                index += 1

        matrix_size = index
        self.rho_matrix = np.zeros((matrix_size, matrix_size), dtype=np.complex128)

    @log_method
    def reset_matrix(self):
        """
        Reset the matrix to zeros.
        """
        self.rho_matrix = self.rho_matrix * 0

    @log_method
    def add_coefficient_from_df(self, df: pd.DataFrame):
        r"""
        Add coefficients to the matrix from a dataframe with columns
        ``index0``, ``index1``, ``coefficient``.
        """
        df = df[["index0", "index1", "coefficient"]].groupby(["index0", "index1"]).sum().reset_index()
        self.rho_matrix[df.index0, df.index1] += df.coefficient
