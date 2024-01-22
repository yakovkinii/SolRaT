import logging

import numpy as np
from typing import Dict

from numpy import pi, sqrt

from core.utility.math import m1p, delta
from core.utility.python import range_inclusive
from core.utility.wigner_3j_6j_9j import w9j, w6j, w3j
from pipeline.two_term_atom.object import (
    Rhos,
    Transitions,
    Term,
    Transition,
    MatrixBuilder,
)


class TwoTermAtom:
    def __init__(self):
        self.rhos = Rhos()
        self.terms = [
            Term(id="2p_2P0.5", l=1, s=0.5, j=0.5, energy=82258.9191133),
            Term(id="2p_2P1.5", l=1, s=0.5, j=1.5, energy=82259.2850014),
            Term(id="3s_2S0.5", l=0, s=0.5, j=0.5, energy=97492.221701),
        ]
        self.terms: Dict[str, Term] = {term.id: term for term in self.terms}

        self.transitions = Transitions()
        self.transitions.add(
            Transition(
                term_upper=self.terms["3s_2S0.5"],
                term_lower=self.terms["2p_2P0.5"],
                einstein_a=0.1,
            )
        )
        self.transitions.add(
            Transition(
                term_upper=self.terms["3s_2S0.5"],
                term_lower=self.terms["2p_2P1.5"],
                einstein_a=0.1,
            )
        )
        self.matrix_builder = MatrixBuilder(terms=list(self.terms.values()))

    @staticmethod
    def j_tensor(k: int, q: int, transition: Transition):
        if k > 2:
            return 0
        if abs(q) > k:
            return 0
        _ = transition
        logging.warning('Using mock radiation tensor.')
        return 1 * delta(k, 0) * delta(q, 0)  # todo

    def add_all_equations(self):
        """
        Loops through all equations.

        Reference: (7.38)
        """
        self.matrix_builder.reset_matrix()

        for term in self.terms.values():
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
                            self.add_single_equation(
                                term=term, k=k, q=q, j=j, j_prime=j_prime
                            )

    def add_single_equation(self, term: Term, k: int, q: int, j: float, j_prime: float):
        """
        Adds single equation to the rho matrix

        Reference: (7.38)

        Expressions like

        d/dt rho(K0, Q0, ...) = sum_{K,Q}(
            A(K, Q, ...) * rho(K, Q, ...)
        )

        should be written as

        for K in <relevant range>:
            for Q in <relevant range>:
                add_coefficient(
                    K0, Q0, ...,
                    K, Q, ...,
                    coefficient = A(K, Q, ...)
                )

        """

        self.matrix_builder.select_equation(
            term=term,
            k=k,
            q=q,
            j=j,
            j_prime=j_prime,
        )

        # Coherence relaxation
        # for k_prime in range_inclusive():
        #     for q_prime in range_inclusive(-k_prime, k_prime):
        #         for j_prime_prime in range_inclusive():
        #             for j_prime_prime_prime in range_inclusive():
        #                 add_coefficient(
        #                     term_id_1=term_id,
        #                     k_1=k_prime,
        #                     q_1=q_prime,
        #                     j_1=j_prime_prime,
        #                     j_prime_1=j_prime_prime_prime,
        #                     coefficient=-2 * pi * 1j * self.n(...),
        #                 )

        # Absorption
        for transition in self.transitions.transitions_from_lower[term.id]:
            term_lower = transition.term_lower
            if term_lower.s != term.s:
                continue
            s = term.s
            l_l = term_lower.l
            for k_l in range_inclusive(k + 2):  # (7.45a, 3j). Kr is always <= 2
                for q_l in range_inclusive(-k_l, k_l):
                    for j_l in range_inclusive(l_l + s):  # (7.45a, 1st 6j)
                        for j_prime_l in range_inclusive(l_l + s):  # (7.45a, 2nd 6j)
                            coefficient = self.t_a(
                                transition=transition,
                                term=term,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                                term_lower=term_lower,
                                k_l=k_l,
                                q_l=q_l,
                                j_l=j_l,
                                j_prime_l=j_prime_l,
                            )
                            self.matrix_builder.add_coefficient(
                                term=term_lower,
                                k=k_l,
                                q=q_l,
                                j=j_l,
                                j_prime=j_prime_l,
                                coefficient=coefficient,
                            )

        # Emission
        for transition in self.transitions.transitions_from_upper[term.id]:
            term_upper = transition.term_upper
            if term_upper.s != term.s:
                continue
            s = term.s
            l_u = term_upper.l
            for k_u in range_inclusive(k + 2):
                for q_u in range_inclusive(-k_u, k_u):
                    for j_u in range_inclusive(l_u + s):
                        for j_prime_u in range_inclusive(l_u + s):
                            coefficient_t_e = self.t_e(
                                transition=transition,
                                term=term,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                                term_upper=term_upper,
                                k_u=k_u,
                                q_u=q_u,
                                j_u=j_u,
                                j_prime_u=j_prime_u,
                            )
                            coefficient_t_s = self.t_s(
                                transition=transition,
                                term=term,
                                k=k,
                                q=q,
                                j=j,
                                j_prime=j_prime,
                                term_upper=term_upper,
                                k_u=k_u,
                                q_u=q_u,
                                j_u=j_u,
                                j_prime_u=j_prime_u,
                            )
                            self.matrix_builder.add_coefficient(
                                term=term_upper,
                                k=k_u,
                                q=q_u,
                                j=j_u,
                                j_prime=j_prime_u,
                                coefficient=coefficient_t_e + coefficient_t_s,
                            )

        # Relaxation
        # for k_prime in range_inclusive():
        #     for q_prime in range_inclusive(-k_prime, k_prime):
        #         for j_prime_prime in range_inclusive():
        #             for j_prime_prime_prime in range_inclusive():
        #                 add_coefficient(
        #                     term_id_1=term_id,
        #                     k_1=k_prime,
        #                     q_1=q_prime,
        #                     j_1=j_prime_prime,
        #                     j_prime_1=j_prime_prime_prime,
        #                     coefficient=-(
        #                         self.r_a(...) + self.r_e(...) + self.r_s(...)
        #                     ),
        #                 )

    def t_a(
        self,
        transition: Transition,
        term: Term,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        term_lower: Term,
        k_l: int,
        q_l: int,
        j_l: float,
        j_prime_l: float,
    ):
        """
        Reference: (7.45a)
        """
        s = term.s
        l = term_lower.l
        l_l = term_lower.l

        if (
            s != term_lower.s
        ):  # This is already treated in add_coefficient(), but adding just in case
            return 0

        m0 = (2 * l_l + 1) * transition.einstein_b_lu
        result = 0
        for k_r in [0, 1, 2]:
            for q_r in range_inclusive(-k_r, k_r):
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

    @staticmethod
    def t_e(
        transition: Transition,
        term: Term,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        term_upper: Term,
        k_u: int,
        q_u: int,
        j_u: float,
        j_prime_u: float,
    ):
        """
        Reference: (7.45b)
        """
        s = term.s
        l = term.l
        l_u = term_upper.l

        if s != term_upper.s:
            return 0
        if k != k_u:
            return 0
        if q != q_u:
            return 0

        m0 = (2 * l_u + 1) * transition.einstein_a
        m1 = sqrt((2 * j + 1) * (2 * j_prime + 1) * (2 * j_u + 1) * (2 * j_prime_u + 1))
        m2 = m1p(1 + k + j_prime + j_prime_u)
        m3 = w6j(j, j_prime, k, j_prime_u, j_u, 1)
        m4 = w6j(l_u, l, 1, j, j_u, s)
        m5 = w6j(l_u, l, 1, j_prime, j_prime_u, s)
        result = m0 * m1 * m2 * m3 * m4 * m5
        return result

    def t_s(
        self,
        transition: Transition,
        term: Term,
        k: int,
        q: int,
        j: float,
        j_prime: float,
        term_upper: Term,
        k_u: int,
        q_u: int,
        j_u: float,
        j_prime_u: float,
    ):
        """
        Reference: (7.45c)
        """
        s = term.s
        l = term.l
        l_u = term_upper.l

        if s != term_upper.s:
            return 0

        m0 = (2 * l_u + 1) * transition.einstein_b_ul
        result = 0
        for k_r in [0, 1, 2]:
            for q_r in range_inclusive(-k_r, k_r):
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
