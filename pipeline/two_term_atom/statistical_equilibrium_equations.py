import logging
from typing import List

import numpy as np

from numpy import pi, sqrt

from core.tensor.atmosphere_parameters import AtmosphereParameters
from core.tensor.radiation_tensor import RadiationTensor
from core.utility.black_body import get_BP
from core.utility.constant import h, c
from core.utility.einstein_coefficients import (
    b_ul_from_a_two_level_atom,
    b_lu_from_b_ul_two_level_atom,
)
from core.utility.generator import nested_loops
from core.utility.math import m1p, delta
from core.utility.python import triangular, intersection, projection, triangular_with_kr
from core.utility.wigner_3j_6j_9j import w9j, w6j, w3j
from pipeline.two_term_atom.matrix_builder import (
    MatrixBuilder,
    Level,
)
from pipeline.two_term_atom.term_registry import TermRegistry
from pipeline.two_term_atom.transition_registry import TransitionRegistry, Transition


class TwoTermAtom:
    def __init__(
        self,
        term_registry: TermRegistry,
        transition_registry: TransitionRegistry,
        atmosphere_parameters: AtmosphereParameters,
        radiation_tensor: RadiationTensor,
    ):
        self.term_registry: TermRegistry = term_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.matrix_builder: MatrixBuilder = MatrixBuilder(
            levels=list(self.term_registry.levels.values())
        )
        self.atmosphere_parameters: AtmosphereParameters = atmosphere_parameters
        self.radiation_tensor: RadiationTensor = radiation_tensor
        self.options: List[str] = []  # Todo

    def add_all_equations(self):
        """
        Loops through all equations.

        Reference: (7.38)
        """
        self.matrix_builder.reset_matrix()

        for level in self.term_registry.levels.values():
            for j in triangular(level.l, level.s):
                for j_prime in triangular(level.l, level.s):
                    for k in triangular(j, j_prime):
                        for q in projection(k):
                            self.matrix_builder.select_equation(
                                level=level,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                            )
                            self.add_coherence_decay(
                                level=level,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                            )
                            self.add_absorption(
                                level=level,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                            )
                            self.add_emission(
                                level=level,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                            )
                            self.add_relaxation(
                                level=level,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                            )

    def add_coherence_decay(
        self, level: Level, k: int, q: int, j: float, j_prime: float
    ):
        logging.info("add_coherence_decay")
        """
        Reference: (7.38)
        """
        for k_prime in triangular(k, 1):
            for q_prime in projection(k_prime):
                for j_prime_prime in intersection(
                    triangular(j, 1), triangular(level.l, level.s)
                ):
                    for j_prime_prime_prime in intersection(
                        triangular(j, 1), triangular(level.l, level.s)
                    ):
                        n = self.n(
                            level=level,
                            k=k,
                            q=q,
                            j=j,
                            j_prime=j_prime,
                            k_prime=k_prime,
                            q_prime=q_prime,
                            j_prime_prime=j_prime_prime,
                            j_prime_prime_prime=j_prime_prime_prime,
                        )
                        self.matrix_builder.add_coefficient(
                            level=level,
                            k=k_prime,
                            q=q_prime,
                            j=j_prime_prime,
                            j_prime=j_prime_prime_prime,
                            coefficient=-2 * pi * 1j * n,
                        )

    def add_absorption(self, level: Level, k: int, q: int, j: float, j_prime: float):
        logging.info("add_absorption")
        """
        Reference: (7.38)
        """
        # Absorption toward selected coherence
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(
                level_upper=level, level_lower=level_lower
            ):
                continue
            for j_l in triangular(level_lower.l, level_lower.s):
                for j_prime_l in triangular(level_lower.l, level_lower.s):
                    for k_l in intersection(
                        triangular_with_kr(k), triangular(j_l, j_prime_l)
                    ):
                        for q_l in projection(k_l):
                            t_a = self.t_a(
                                level=level,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                                level_lower=level_lower,
                                k_l=k_l,
                                q_l=q_l,
                                j_l=j_l,
                                j_prime_l=j_prime_l,
                            )
                            self.matrix_builder.add_coefficient(
                                level=level_lower,
                                k=k_l,
                                q=q_l,
                                j=j_l,
                                j_prime=j_prime_l,
                                coefficient=t_a,
                            )

    def add_emission(self, level: Level, k: int, q: int, j: float, j_prime: float):
        logging.info("add_emission")
        """
        Reference: (7.38)
        """
        # Emission toward selected coherence
        for level_upper in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(
                level_upper=level_upper, level_lower=level
            ):
                continue
            for j_u in triangular(level_upper.l, level_upper.s):
                for j_prime_u in triangular(level_upper.l, level_upper.s):
                    for k_u in intersection(
                        triangular_with_kr(k), triangular(j_u, j_prime_u)
                    ):
                        for q_u in projection(k_u):
                            t_e = self.t_e(
                                level=level,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                                level_upper=level_upper,
                                k_u=k_u,
                                q_u=q_u,
                                j_u=j_u,
                                j_prime_u=j_prime_u,
                            )
                            t_s = self.t_s(
                                level=level,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                                level_upper=level_upper,
                                k_u=k_u,
                                q_u=q_u,
                                j_u=j_u,
                                j_prime_u=j_prime_u,
                            )
                            self.matrix_builder.add_coefficient(
                                level=level_upper,
                                k=k_u,
                                q=q_u,
                                j=j_u,
                                j_prime=j_prime_u,
                                coefficient=t_e + t_s,
                            )

    def add_relaxation(self, level: Level, k: int, q: int, j: float, j_prime: float):
        logging.info("add_relaxation")
        """
        Reference: (7.38)
        """
        # Relaxation from selected coherence
        for k_prime, q_prime, j_prime_prime, j_prime_prime_prime in nested_loops(
            {
                "k_prime": f"triangular({j}, {j_prime})",
                "q_prime": f"projection(k_prime)",
                "j_prime_prime": f"triangular({level.l}, {level.s})",
                "j_prime_prime_prime": f"triangular({level.l}, {level.s})",
            }
        ):
            r_a = self.r_a(
                level=level,
                k=k,
                q=q,
                j=j,
                j_prime=j_prime,
                k_prime=k_prime,
                q_prime=q_prime,
                j_prime_prime=j_prime_prime,
                j_prime_prime_prime=j_prime_prime_prime,
            )
            r_e = self.r_e(
                level=level,
                k=k,
                q=q,
                j=j,
                j_prime=j_prime,
                k_prime=k_prime,
                q_prime=q_prime,
                j_prime_prime=j_prime_prime,
                j_prime_prime_prime=j_prime_prime_prime,
            )
            if "disable_r_s" in self.options:
                r_s = 0
            else:
                r_s = self.r_s(
                    level=level,
                    k=k,
                    q=q,
                    j=j,
                    j_prime=j_prime,
                    k_prime=k_prime,
                    q_prime=q_prime,
                    j_prime_prime=j_prime_prime,
                    j_prime_prime_prime=j_prime_prime_prime,
                )
            self.matrix_builder.add_coefficient(
                level=level,
                k=k_prime,
                q=q_prime,
                j=j_prime_prime,
                j_prime=j_prime_prime_prime,
                coefficient=-(r_a + r_e + r_s),
            )

    def r_a(
        self,
        level: Level,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        k_prime: int,
        q_prime: int,
        j_prime_prime: float,
        j_prime_prime_prime: float,
    ):
        l = level.l
        s = level.s
        result = 0
        for level_upper in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(
                level_upper=level_upper, level_lower=level
            ):
                continue
            transition = self.transition_registry.get_transition(
                level_upper=level_upper, level_lower=level
            )
            m0 = (2 * l + 1) * transition.einstein_b_lu
            l_u = level_upper.l
            for k_r in [0, 1, 2]:
                for q_r in projection(k_r):  # todo: no need to sum: q_r = q_prime - q
                    m1 = sqrt(3 * (2 * k + 1) * (2 * k_prime + 1) * (2 * k_r + 1))
                    m2 = m1p(1 + l_u - s + j + q_prime)
                    m3 = w6j(l, l, k_r, 1, 1, l_u) * w3j(
                        k, k_prime, k_r, q, -q_prime, q_r
                    )
                    m4 = 0.5 * self.radiation_tensor.get(
                        transition=transition, k=k_r, q=q_r
                    )
                    a1 = delta(j, j_prime_prime) * sqrt(
                        (2 * j_prime + 1) * (2 * j_prime_prime_prime + 1)
                    )
                    a2 = w6j(l, l, k_r, j_prime_prime_prime, j_prime, s)
                    a3 = w6j(k, k_prime, k_r, j_prime_prime_prime, j_prime, j)
                    b1 = delta(j_prime, j_prime_prime_prime) * sqrt(
                        (2 * j + 1) * (2 * j_prime_prime + 1)
                    )
                    b2 = m1p(j_prime_prime - j_prime + k + k_prime + k_r)
                    b3 = w6j(l, l, k_r, j_prime_prime, j, s)
                    b4 = w6j(k, k_prime, k_r, j_prime_prime, j, j_prime)
                    result += (
                        m0 * m1 * m2 * m3 * m4 * (a1 * a2 * a3 + b1 * b2 * b3 * b4)
                    )
        return result

    def r_e(
        self,
        level: Level,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        k_prime: int,
        q_prime: int,
        j_prime_prime: float,
        j_prime_prime_prime: float,
    ):
        result = 0
        m0 = (
            delta(k, k_prime)
            * delta(q, q_prime)
            * delta(j, j_prime_prime)
            * delta(j_prime, j_prime_prime_prime)
        )
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(
                level_upper=level, level_lower=level_lower
            ):
                continue
            transition = self.transition_registry.get_transition(
                level_upper=level, level_lower=level_lower
            )
            result += m0 * transition.einstein_a_ul
        return result

    def r_s(
        self,
        level: Level,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        k_prime: int,
        q_prime: int,
        j_prime_prime: float,
        j_prime_prime_prime: float,
    ):
        # Todo all relaxation rates can be simplified (7.46)
        l = level.l
        s = level.s
        result = 0
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(
                level_upper=level, level_lower=level_lower
            ):
                continue
            transition = self.transition_registry.get_transition(
                level_upper=level, level_lower=level_lower
            )
            m0 = (2 * l + 1) * transition.einstein_b_ul
            l_l = level_lower.l
            for k_r in [0, 1, 2]:
                for q_r in projection(k_r):  # todo: no need to sum: q_r = q_prime - q
                    m1 = sqrt(3 * (2 * k + 1) * (2 * k_prime + 1) * (2 * k_r + 1))
                    m2 = m1p(1 + l_l - s + j + k_r + q_prime)
                    m3 = w6j(l, l, k_r, 1, 1, l_l) * w3j(
                        k, k_prime, k_r, q, -q_prime, q_r
                    )
                    m4 = 0.5 * self.radiation_tensor.get(
                        transition=transition, k=k_r, q=q_r
                    )
                    a1 = delta(j, j_prime_prime) * sqrt(
                        (2 * j_prime + 1) * (2 * j_prime_prime_prime + 1)
                    )
                    a2 = w6j(l, l, k_r, j_prime_prime_prime, j_prime, s)
                    a3 = w6j(k, k_prime, k_r, j_prime_prime_prime, j_prime, j)
                    b1 = delta(j_prime, j_prime_prime_prime) * sqrt(
                        (2 * j + 1) * (2 * j_prime_prime + 1)
                    )
                    b2 = m1p(j_prime_prime - j_prime + k + k_prime + k_r)
                    b3 = w6j(l, l, k_r, j_prime_prime, j, s)
                    b4 = w6j(k, k_prime, k_r, j_prime_prime, j, j_prime)
                    result += (
                        m0 * m1 * m2 * m3 * m4 * (a1 * a2 * a3 + b1 * b2 * b3 * b4)
                    )
        return result

    @staticmethod
    def gamma(
        level: Level,
        j: float,
        j_prime: float,
    ):
        s = level.s
        l = level.l
        result = delta(j, j_prime) * sqrt(j * (j + 1) * (2 * j + 1))
        result += (
            m1p(1 + l + s + j)
            * sqrt((2 * j + 1) * (2 * j_prime + 1) * s * (s + 1) * (2 * s + 1))
            * w6j(j, j_prime, 1, s, s, l)
        )
        return result

    def n(
        self,
        level: Level,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        k_prime: int,
        q_prime: int,
        j_prime_prime: float,
        j_prime_prime_prime: float,
    ):
        """
        Reference: (7.41)
        """
        term = self.term_registry.get_term(level=level, j=j)
        term_prime = self.term_registry.get_term(level=level, j=j_prime)
        nu = (term.energy - term_prime.energy) / h

        result = (
            delta(k, k_prime)
            * delta(q, q_prime)
            * delta(j, j_prime_prime)
            * delta(j_prime, j_prime_prime_prime)
            * nu
        )

        result += (
            delta(q, q_prime)
            * self.atmosphere_parameters.nu_larmor
            * m1p(j + j_prime - q)
            * sqrt((2 * k + 1) * (2 * k_prime + 1))
            * w3j(k, k_prime, 1, -q, q, 0)
            * (
                delta(j_prime, j_prime_prime_prime)
                * self.gamma(level=level, j=j, j_prime=j_prime_prime)
                * w6j(k, k_prime, 1, j_prime_prime, j, j_prime)
                + delta(j, j_prime_prime)
                * self.gamma(level=level, j=j_prime_prime_prime, j_prime=j_prime)
                * w6j(k, k_prime, 1, j_prime_prime_prime, j_prime, j)
            )
        )
        return result

    def t_a(
        self,
        level: Level,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        level_lower: Level,
        k_l: int,
        q_l: int,
        j_l: float,
        j_prime_l: float,
    ):
        """
        Reference: (7.45a)
        """
        s = level.s
        l = level.l
        l_l = level_lower.l

        assert s == level_lower.s

        transition = self.transition_registry.get_transition(
            level_upper=level, level_lower=level_lower
        )
        m0 = (2 * l_l + 1) * transition.einstein_b_lu
        result = 0
        for k_r in [0, 1, 2]:
            for q_r in projection(k_r):  # todo: no need to sum: q_r = q_l - q
                m1 = sqrt(
                    3
                    * (2 * j + 1)
                    * (2 * j_prime + 1)
                    * (2 * j_l + 1)
                    * (2 * j_prime_l + 1)
                    * (2 * k + 1)
                    * (2 * k_l + 1)
                    * (2 * k_r + 1)
                )
                m2 = m1p(k_l + q_l + j_prime_l - j_l)
                m3 = w9j(j, j_l, 1, j_prime, j_prime_l, 1, k, k_l, k_r)
                m4 = w6j(l, l_l, 1, j_l, j, s)
                m5 = w6j(l, l_l, 1, j_prime_l, j_prime, s)
                m6 = w3j(k, k_l, k_r, -q, q_l, -q_r)
                m7 = self.radiation_tensor.get(transition=transition, k=k_r, q=q_r)
                result += m0 * m1 * m2 * m3 * m4 * m5 * m6 * m7
        return result

    def t_e(
        self,
        level: Level,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        level_upper: Level,
        k_u: int,
        q_u: int,
        j_u: float,
        j_prime_u: float,
    ):
        """
        Reference: (7.45b)
        """
        s = level.s
        l = level.l
        l_u = level_upper.l

        assert s == level_upper.s

        if k != k_u:
            return 0
        if q != q_u:
            return 0

        transition = self.transition_registry.get_transition(
            level_upper=level_upper, level_lower=level
        )
        m0 = (2 * l_u + 1) * transition.einstein_a_ul
        m1 = sqrt((2 * j + 1) * (2 * j_prime + 1) * (2 * j_u + 1) * (2 * j_prime_u + 1))
        m2 = m1p(1 + k + j_prime + j_prime_u)
        m3 = w6j(j, j_prime, k, j_prime_u, j_u, 1)
        m4 = w6j(l_u, l, 1, j, j_u, s)
        m5 = w6j(l_u, l, 1, j_prime, j_prime_u, s)
        result = m0 * m1 * m2 * m3 * m4 * m5
        return result

    def t_s(
        self,
        level: Level,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        level_upper: Level,
        k_u: int,
        q_u: int,
        j_u: float,
        j_prime_u: float,
    ):
        """
        Reference: (7.45c)
        """
        s = level.s
        l = level.l
        l_u = level_upper.l

        assert s == level_upper.s

        transition = self.transition_registry.get_transition(
            level_upper=level_upper, level_lower=level
        )

        m0 = (2 * l_u + 1) * transition.einstein_b_ul
        result = 0
        for k_r in [0, 1, 2]:
            for q_r in projection(k_r):  # todo: no need to sum: q_r = q_u - q
                m1 = sqrt(
                    3
                    * (2 * j + 1)
                    * (2 * j_prime + 1)
                    * (2 * j_u + 1)
                    * (2 * j_prime_u + 1)
                    * (2 * k + 1)
                    * (2 * k_u + 1)
                    * (2 * k_r + 1)
                )
                m2 = m1p(k_r + k_u + q_u + j_prime_u - j_u)
                m3 = w9j(j, j_u, 1, j_prime, j_prime_u, 1, k, k_u, k_r)
                m4 = w6j(l_u, l, 1, j, j_u, s)
                m5 = w6j(l_u, l, 1, j_prime, j_prime_u, s)
                m6 = w3j(k, k_u, k_r, -q, q_u, -q_r)
                m7 = self.radiation_tensor.get(transition=transition, k=k_r, q=q_r)
                result += m0 * m1 * m2 * m3 * m4 * m5 * m6 * m7
        return result

    def get_solution_direct(self):
        # A x = 0
        # let x[0] = 1 (it is always rho_0_0 of some kind, so it is unlikely to be exactly zero)
        # A[1:] x = 0
        # A[1:, 1:] x[1:] = -A[1:, 0]
        sol = np.linalg.solve(
            self.matrix_builder.rho_matrix[1:, 1:],
            -self.matrix_builder.rho_matrix[1:, 0],
        )
        sol = np.insert(sol, 0, 1.0, 0)

        # Sum sqrt(2J+1) rho00(J, J) = 1
        trace = sum(
            [
                sol[index] * weight
                for (index, weight) in zip(
                    self.matrix_builder.trace_indexes, self.matrix_builder.trace_weights
                )
            ]
        )
        solution_vector = sol / trace
        return solution_vector
