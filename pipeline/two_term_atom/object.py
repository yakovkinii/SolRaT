from typing import List, Dict

import numpy as np
from numpy import sqrt, exp, pi

from core.utility.constant import h
from core.utility.einstein_coefficients import (
    b_lu_from_b_ul_two_level_atom,
    b_ul_from_a_two_level_atom,
)
from core.utility.python import range_inclusive


class Term:
    def __init__(self, id: str, l: float, s: float, j: float, energy: float):
        self.id: str = id
        self.l: float = l
        self.s: float = s
        self.j: float = j
        self.energy: float = energy


class Transition:
    def __init__(self, term_upper: Term, term_lower: Term, einstein_a: float):
        self.term_upper: Term = term_upper
        self.term_lower: Term = term_lower
        self.einstein_a: float = einstein_a
        self.nu: float = (self.term_upper.energy - self.term_lower.energy) / h
        self.einstein_b_ul: float = b_ul_from_a_two_level_atom(
            self.einstein_a, nu=self.nu
        )
        self.einstein_b_lu: float = b_lu_from_b_ul_two_level_atom(
            self.einstein_b_ul, j_u=self.term_upper.j, j_l=self.term_lower.j
        )


class Transitions:
    def __init__(self):
        """
        This class helps to manage which transitions are fou upper levels and which are from lower.
        """
        self.transitions_from_upper: Dict[str, List[Transition]] = dict()
        self.transitions_from_lower: Dict[str, List[Transition]] = dict()
        self.transitions_all: Dict[str, List[Transition]] = dict()

    def add(self, transition: Transition):
        if transition.term_upper.id in self.transitions_all.keys():
            self.transitions_all[transition.term_upper.id].append(transition)
        else:
            self.transitions_all[transition.term_upper.id] = [transition]

        if transition.term_lower.id in self.transitions_all.keys():
            self.transitions_all[transition.term_lower.id].append(transition)
        else:
            self.transitions_all[transition.term_lower.id] = [transition]

        if transition.term_upper.id in self.transitions_from_upper.keys():
            self.transitions_from_upper[transition.term_upper.id].append(transition)
        else:
            self.transitions_from_upper[transition.term_upper.id] = [transition]

        if transition.term_lower.id in self.transitions_from_lower.keys():
            self.transitions_from_lower[transition.term_lower.id].append(transition)
        else:
            self.transitions_from_lower[transition.term_lower.id] = [transition]


class Rhos:
    def __init__(self):
        """
        rho{beta L S, K, Q}(J, J')

        rho{term_id, K, Q}(J, J')
        term_id defines beta L S

        transitions: id -> id
        """
        self.data = dict()

    def set(self, term_id, K, Q, J1, J2, value):
        self.data[(term_id, K, Q, J1, J2)] = value

    def __call__(self, term_id, K, Q, J1, J2):
        return self.data[(term_id, K, Q, J1, J2)]


class MatrixBuilder:
    def __init__(self, terms: List[Term]):
        """
        This class helps to build the matrix for rhos.
        All possible rhos are defined by terms.
        """
        # Create mapping [term_id, K, Q, J, J'] <-> matrix index
        self.index_to_parameters = dict()
        self.parameters_to_index = dict()

        index = 0
        for term in terms:
            # J, J' = L+S ... |L-S|
            j_max = term.l + term.s
            j_min = abs(term.l - term.s)
            for j in range_inclusive(j_min, j_max):
                for j_prime in range_inclusive(j_min, j_max):
                    # K = J + J' ... |J - J'|
                    k_max = j + j_prime
                    k_min = abs(j - j_prime)
                    for k in range_inclusive(k_min, k_max):
                        # Q = -K ... K
                        for q in range_inclusive(-k, k):
                            self.parameters_to_index[
                                (term.id, k, q, j, j_prime)
                            ] = index
                            self.index_to_parameters[index] = (
                                term.id,
                                k,
                                q,
                                j,
                                j_prime,
                            )
                            index += 1
        # create the matrix
        rho_matrix_size = index
        self.rho_matrix = np.zeros(
            (rho_matrix_size, rho_matrix_size), dtype=np.complex128
        )

        self.selected_equation = None

    def reset_matrix(self):
        self.rho_matrix = self.rho_matrix * 0

    def select_equation(
        self,
        term: Term,
        k: int,
        q: int,
        j: float,
        j_prime: float,
    ):
        """
        Selects an equation (row in matrix) to add coefficients to.
        """
        self.selected_equation = (term.id, k, q, j, j_prime)

    def add_coefficient(
        self,
        term: Term,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        coefficient: float,
    ):
        """
        Adds a coefficient to the selected equation.
        """
        index0 = self.parameters_to_index[self.selected_equation]
        index1 = self.parameters_to_index[(term.id, k, q, j, j_prime)]
        self.rho_matrix[index0, index1] += coefficient
