import logging

import numpy as np

from numpy import pi, sqrt

from core.utility.constant import h
from core.utility.math import m1p, delta
from core.utility.misc import nu_larmor
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
        atmosphere_parameters=None,
        radiation_tensor=None,
    ):
        self.term_registry: TermRegistry = term_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.matrix_builder = MatrixBuilder(
            levels=list(self.term_registry.levels.values())
        )
        self.nu_larmor = nu_larmor(magnetic_field_gauss=0)  # todo remove

    @staticmethod
    def j_tensor(k: int, q: int, transition: Transition):  # todo remove
        if k > 2:
            return 0
        if abs(q) > k:
            return 0
        _ = transition
        return 0 * delta(k, 0) * delta(q, 0)  # todo
        # return 1e2

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
        """
        Reference: (7.38)
        """
        # Relaxation from selected coherence
        for k_prime in triangular(j, j_prime):
            for q_prime in projection(k_prime):
                for j_prime_prime in triangular(level.l, level.s):  # todo
                    for j_prime_prime_prime in triangular(level.l, level.s):  # todo
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
                for q_r in projection(k_r):
                    m1 = sqrt(3 * (2 * k + 1) * (2 * k_prime + 1) * (2 * k_r + 1))
                    m2 = m1p(1 + l_u - s + j + q_prime)
                    m3 = w6j(l, l, k_r, 1, 1, l_u) * w3j(
                        k, k_prime, k_r, q, -q_prime, q_r
                    )
                    m4 = 0.5 * self.j_tensor(k=k_r, q=q_r, transition=transition)
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
                for q_r in projection(k_r):
                    m1 = sqrt(3 * (2 * k + 1) * (2 * k_prime + 1) * (2 * k_r + 1))
                    m2 = m1p(1 + l_l - s + j + k_r + q_prime)
                    m3 = w6j(l, l, k_r, 1, 1, l_l) * w3j(
                        k, k_prime, k_r, q, -q_prime, q_r
                    )
                    m4 = 0.5 * self.j_tensor(k=k_r, q=q_r, transition=transition)
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
            * self.nu_larmor
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
            for q_r in projection(k_r):
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
                m7 = self.j_tensor(k=k_r, q=q_r, transition=transition)
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
            for q_r in projection(k_r):
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
                m7 = self.j_tensor(k=k_r, q=q_r, transition=transition)
                result += m0 * m1 * m2 * m3 * m4 * m5 * m6 * m7
        return result

    # def r_a(
    #     self,
    #     term_id,
    #     k,
    #     q,
    #     j,
    #     j_prime,
    #     k_prime,
    #     q_prime,
    #     j_prime_prime,
    #     j_prime_prime_prime,
    # ):
    #     """
    #     (7.46a)
    #     """
    #     s = self.terms[term_id].s
    #     l = self.terms[term_id].l
    #
    #     _a0 = 2 * l + 1
    #     result = 0
    #     for term_id_upper in self.transitions.transitions_from_upper[term_id]:
    #         l_u = self.terms[term_id_upper].l
    #         _a1 = self.einstein_b_lu(lower_term_id=term_id, upper_term_id=term_id_upper)
    #         for k_r in range_inclusive():
    #             for q_r in range_inclusive(-k_r, k_r):
    #                 _a2 = sqrt(3 * (2 * k + 1) * (2 * k_prime + 1) * (2 * k_r + 1))
    #                 _a3 = m1p(1 + l_u - s + j + q_prime)
    #                 _a4 = w6j(l, l, k_r, 1, 1, l_u)
    #                 _a5 = w3j(k, k_prime, k_r, q, -q_prime, q_r)
    #                 _a6 = 0.5
    #                 _a7 = self.j_tensor(
    #                     k=k_r, q=q_r, lower_term_id=term_id, upper_term_id=term_id_upper
    #                 )
    #
    #                 _b0 = delta(j, j_prime_prime)
    #                 _b1 = sqrt((2 * j_prime + 1) * (2 * j_prime_prime_prime + 1))
    #                 _b2 = w6j(l, l, k_r, j_prime_prime_prime, j_prime, s)
    #                 _b3 = w6j(k, k_prime, k_r, j_prime_prime_prime, j_prime, j)
    #
    #                 _c0 = delta(j_prime, j_prime_prime_prime)
    #                 _c1 = sqrt((2 * j + 1) * (2 * j_prime_prime + 1))
    #                 _c2 = m1p(j_prime_prime - j_prime + k + k_prime + k_r)
    #                 _c3 = w6j(l, l, k_r, j_prime_prime, j, s)
    #                 _c4 = w6j(k, k_prime, k_r, j_prime_prime, j, j_prime)
    #
    #                 _a = _a0 * _a1 * _a2 * _a3 * _a4 * _a5 * _a6 * _a7
    #                 _b = _b0 * _b1 * _b2 * _b3
    #                 _c = _c0 * _c1 * _c2 * _c3 * _c4
    #
    #                 result += _a * (_b + _c)
    #
    #     return result


if __name__ == "__main__":
    term_registry = TermRegistry()
    term_registry.register_term(
        beta="1s",
        l=0,
        s=0,
        j=0,
        energy=80000,
    )
    term_registry.register_term(
        beta="2p",
        l=1,
        s=0,
        j=1,
        energy=82000,
    )
    term_registry.validate()

    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        level_upper=term_registry.get_level(beta="2p", l=1, s=0),
        level_lower=term_registry.get_level(beta="1s", l=0, s=0),
        einstein_a_ul=0.1,
        einstein_b_ul=0.1,
        einstein_b_lu=0.1,
    )
    atom = TwoTermAtom(
        term_registry=term_registry, transition_registry=transition_registry
    )
    atom.add_all_equations()

    rho_matrix = atom.matrix_builder.rho_matrix
    rho_matrix_abs = np.abs(rho_matrix)
    import pandas as pd

    mat = pd.DataFrame(rho_matrix_abs)
    mat.index = atom.matrix_builder.coherence_id_to_index.keys()
    mat.columns = atom.matrix_builder.coherence_id_to_index.keys()
    U, S, VT = np.linalg.svd(atom.matrix_builder.rho_matrix)
    index_min_singular_value = np.argmin(S)

    # The corresponding vector in V^T (hence, V since we're using Python) is our solution
    solution_vector = VT[index_min_singular_value, :]

    print("Solution vector: ", solution_vector)
    ...
