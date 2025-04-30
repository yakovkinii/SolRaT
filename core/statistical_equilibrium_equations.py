import logging
from typing import List

import numpy as np
from numpy import pi, sqrt

from core.base.generator import multiply, n_proj, nested_loops
from core.base.math import delta, m1p, δ
from core.base.python import intersection, projection, triangular, triangular_with_kr
from core.matrix_builder import Level, MatrixBuilder, Rho
from core.object.atmosphere_parameters import AtmosphereParameters
from core.object.radiation_tensor import RadiationTensor
from core.terms_levels_transitions.term_registry import TermRegistry
from core.terms_levels_transitions.transition_registry import TransitionRegistry
from core.utility.constant import h
from core.utility.wigner_3j_6j_9j import w9j, wigner_3j, wigner_6j


class TwoTermAtom:
    def __init__(
        self,
        term_registry: TermRegistry,
        transition_registry: TransitionRegistry,
        atmosphere_parameters: AtmosphereParameters,
        radiation_tensor: RadiationTensor,
        n_frequencies: int,
        disable_r_s: bool = False,
    ):
        self.term_registry: TermRegistry = term_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.matrix_builder: MatrixBuilder = MatrixBuilder(
            levels=list(self.term_registry.levels.values()), n_frequencies=n_frequencies
        )
        self.atmosphere_parameters: AtmosphereParameters = atmosphere_parameters
        self.radiation_tensor: RadiationTensor = radiation_tensor
        self.disable_r_s = disable_r_s

    def add_all_equations(self):
        """
        Loops through all equations.

        Reference: (7.38)
        """
        self.matrix_builder.reset_matrix()
        for level in self.term_registry.levels.values():
            for J, Jʹ, K, Q in nested_loops(
                J=f"triangular({level.l}, {level.s})",
                Jʹ=f"triangular({level.l}, {level.s})",
                K=f"triangular(J, Jʹ)",
                Q="projection(K)",
            ):
                self.matrix_builder.select_equation(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.add_coherence_decay(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.add_absorption(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.add_emission(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.add_relaxation(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)

    def add_coherence_decay(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        logging.info("add_coherence_decay")
        """
        Reference: (7.38)
        """
        for Kʹ, Qʹ, Jʹʹ, Jʹʹʹ in nested_loops(
            Kʹ=f"triangular({K}, 1)",
            Qʹ=f"projection(Kʹ)",
            Jʹʹ=f"intersection(triangular({J}, 1), triangular({level.l}, {level.s}))",
            Jʹʹʹ=f"intersection(triangular({J}, 1), triangular({level.l}, {level.s}))",
        ):
            n = self.n(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            self.matrix_builder.add_coefficient(level=level, K=Kʹ, Q=Qʹ, J=Jʹʹ, Jʹ=Jʹʹʹ, coefficient=-2 * pi * 1j * n)

    def add_absorption(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        logging.info("add_absorption")
        """
        Reference: (7.38)
        """
        # Absorption toward selected coherence
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level, level_lower=level_lower):
                continue

            for Jl, Jʹl, Kl, Ql in nested_loops(
                Jl=f"triangular({level_lower.l}, {level_lower.s})",
                Jʹl=f"triangular({level_lower.l}, {level_lower.s})",
                Kl=f"intersection(triangular_with_kr({K}), triangular(Jl, Jʹl))",
                Ql=f"projection(Kl)",
            ):
                t_a = self.t_a(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, level_lower=level_lower, Kl=Kl, Ql=Ql, Jl=Jl, Jʹl=Jʹl)
                self.matrix_builder.add_coefficient(level=level_lower, K=Kl, Q=Ql, J=Jl, Jʹ=Jʹl, coefficient=t_a)

    def add_emission(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        logging.info("add_emission")
        """
        Reference: (7.38)
        """
        # Emission toward selected coherence
        for level_upper in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level_upper, level_lower=level):
                continue

            for Ju, Jʹu, Ku, Qu in nested_loops(
                Ju=f"triangular({level_upper.l}, {level_upper.s})",
                Jʹu=f"triangular({level_upper.l}, {level_upper.s})",
                Ku=f"intersection(triangular_with_kr({K}), triangular(Ju, Jʹu))",
                Qu=f"projection(Ku)",
            ):
                t_e = self.t_e(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, level_upper=level_upper, Ku=Ku, Qu=Qu, Ju=Ju, Jʹu=Jʹu)
                t_s = self.t_s(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, level_upper=level_upper, Ku=Ku, Qu=Qu, Ju=Ju, Jʹu=Jʹu)
                self.matrix_builder.add_coefficient(level=level_upper, K=Ku, Q=Qu, J=Ju, Jʹ=Jʹu, coefficient=t_e + t_s)

    def add_relaxation(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        logging.info("add_relaxation")
        """
        Reference: (7.38)
        """
        # Relaxation from selected coherence
        for Kʹ, Qʹ, Jʹʹ, Jʹʹʹ in nested_loops(
            Kʹ=f"triangular({J}, {Jʹ})",
            Qʹ=f"projection(Kʹ)",
            Jʹʹ=f"triangular({level.l}, {level.s})",
            Jʹʹʹ=f"triangular({level.l}, {level.s})",
        ):
            r_a = self.r_a(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            r_e = self.r_e(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            if self.disable_r_s:
                r_s = 0
            else:
                r_s = self.r_s(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            self.matrix_builder.add_coefficient(level=level, K=Kʹ, Q=Qʹ, J=Jʹʹ, Jʹ=Jʹʹʹ, coefficient=-(r_a + r_e + r_s))

    def r_a(self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float):
        l = level.l
        s = level.s
        result = 0
        for level_upper in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level_upper, level_lower=level):
                continue
            transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)
            m0 = (2 * l + 1) * transition.einstein_b_lu
            l_u = level_upper.l
            for k_r in [0, 1, 2]:
                for q_r in projection(k_r):  # todo: no need to sum: q_r = q_prime - q
                    m1 = sqrt(3 * (2 * K + 1) * (2 * Kʹ + 1) * (2 * k_r + 1))
                    m2 = m1p(1 + l_u - s + J + Qʹ)
                    m3 = wigner_6j(l, l, k_r, 1, 1, l_u) * wigner_3j(K, Kʹ, k_r, Q, -Qʹ, q_r)
                    m4 = 0.5 * self.radiation_tensor.get(transition=transition, k=k_r, q=q_r)
                    a1 = delta(J, Jʹʹ) * sqrt((2 * Jʹ + 1) * (2 * Jʹʹʹ + 1))
                    a2 = wigner_6j(l, l, k_r, Jʹʹʹ, Jʹ, s)
                    a3 = wigner_6j(K, Kʹ, k_r, Jʹʹʹ, Jʹ, J)
                    b1 = delta(Jʹ, Jʹʹʹ) * sqrt((2 * J + 1) * (2 * Jʹʹ + 1))
                    b2 = m1p(Jʹʹ - Jʹ + K + Kʹ + k_r)
                    b3 = wigner_6j(l, l, k_r, Jʹʹ, J, s)
                    b4 = wigner_6j(K, Kʹ, k_r, Jʹʹ, J, Jʹ)
                    result += m0 * m1 * m2 * m3 * m4 * (a1 * a2 * a3 + b1 * b2 * b3 * b4)
        return result

    def r_e(self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float):
        result = 0
        m0 = delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ)
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level, level_lower=level_lower):
                continue
            transition = self.transition_registry.get_transition(level_upper=level, level_lower=level_lower)
            result += m0 * transition.einstein_a_ul
        return result

    def r_s(self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float):
        # Todo all relaxation rates can be simplified (7.46)
        l = level.l
        s = level.s
        result = 0
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level, level_lower=level_lower):
                continue
            transition = self.transition_registry.get_transition(level_upper=level, level_lower=level_lower)
            m0 = (2 * l + 1) * transition.einstein_b_ul
            l_l = level_lower.l
            for k_r in [0, 1, 2]:
                for q_r in projection(k_r):  # todo: no need to sum: q_r = q_prime - q
                    m1 = sqrt(3 * (2 * K + 1) * (2 * Kʹ + 1) * (2 * k_r + 1))
                    m2 = m1p(1 + l_l - s + J + k_r + Qʹ)
                    m3 = wigner_6j(l, l, k_r, 1, 1, l_l) * wigner_3j(K, Kʹ, k_r, Q, -Qʹ, q_r)
                    m4 = 0.5 * self.radiation_tensor.get(transition=transition, k=k_r, q=q_r)
                    a1 = delta(J, Jʹʹ) * sqrt((2 * Jʹ + 1) * (2 * Jʹʹʹ + 1))
                    a2 = wigner_6j(l, l, k_r, Jʹʹʹ, Jʹ, s)
                    a3 = wigner_6j(K, Kʹ, k_r, Jʹʹʹ, Jʹ, J)
                    b1 = delta(Jʹ, Jʹʹʹ) * sqrt((2 * J + 1) * (2 * Jʹʹ + 1))
                    b2 = m1p(Jʹʹ - Jʹ + K + Kʹ + k_r)
                    b3 = wigner_6j(l, l, k_r, Jʹʹ, J, s)
                    b4 = wigner_6j(K, Kʹ, k_r, Jʹʹ, J, Jʹ)
                    result += m0 * m1 * m2 * m3 * m4 * (a1 * a2 * a3 + b1 * b2 * b3 * b4)
        return result

    @staticmethod
    def gamma(level: Level, j: float, j_prime: float):
        s = level.s
        l = level.l
        result = delta(j, j_prime) * sqrt(j * (j + 1) * (2 * j + 1))
        result += (
            m1p(1 + l + s + j)
            * sqrt((2 * j + 1) * (2 * j_prime + 1) * s * (s + 1) * (2 * s + 1))
            * wigner_6j(j, j_prime, 1, s, s, l)
        )
        return result

    def n(self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float):
        """
        Reference: (7.41)
        """
        term = self.term_registry.get_term(level=level, j=J)
        term_prime = self.term_registry.get_term(level=level, j=Jʹ)
        nu = (term.energy_cmm1 - term_prime.energy_cmm1) / h

        result = delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ) * nu

        result += (
            delta(Q, Qʹ)
            * self.atmosphere_parameters.nu_larmor
            * m1p(J + Jʹ - Q)
            * sqrt((2 * K + 1) * (2 * Kʹ + 1))
            * wigner_3j(K, Kʹ, 1, -Q, Q, 0)
            * (
                delta(Jʹ, Jʹʹʹ) * self.gamma(level=level, j=J, j_prime=Jʹʹ) * wigner_6j(K, Kʹ, 1, Jʹʹ, J, Jʹ)
                + delta(J, Jʹʹ) * self.gamma(level=level, j=Jʹʹʹ, j_prime=Jʹ) * wigner_6j(K, Kʹ, 1, Jʹʹʹ, Jʹ, J)
            )
        )
        return result

    def t_a(
        self,
        level: Level,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        level_lower: Level,
        Kl: int,
        Ql: int,
        Jl: float,
        Jʹl: float,
    ):
        """
        Reference: (7.45a)
        """
        s = level.s
        l = level.l
        l_l = level_lower.l

        assert s == level_lower.s

        transition = self.transition_registry.get_transition(level_upper=level, level_lower=level_lower)
        m0 = (2 * l_l + 1) * transition.einstein_b_lu
        result = 0
        for k_r in [0, 1, 2]:
            for q_r in projection(k_r):  # todo: no need to sum: q_r = q_l - q
                # q_r = Ql - Q  # Todo this errors out, need to investigate
                m1 = sqrt(
                    3
                    * (2 * J + 1)
                    * (2 * Jʹ + 1)
                    * (2 * Jl + 1)
                    * (2 * Jʹl + 1)
                    * (2 * K + 1)
                    * (2 * Kl + 1)
                    * (2 * k_r + 1)
                )
                m2 = m1p(Kl + Ql + Jʹl - Jl)
                m3 = w9j(J, Jl, 1, Jʹ, Jʹl, 1, K, Kl, k_r)
                m4 = wigner_6j(l, l_l, 1, Jl, J, s)
                m5 = wigner_6j(l, l_l, 1, Jʹl, Jʹ, s)
                m6 = wigner_3j(K, Kl, k_r, -Q, Ql, -q_r)
                m7 = self.radiation_tensor.get(transition=transition, k=k_r, q=q_r)
                result += m0 * m1 * m2 * m3 * m4 * m5 * m6 * m7
        return result

    def t_e(
        self,
        level: Level,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        level_upper: Level,
        Ku: int,
        Qu: int,
        Ju: float,
        Jʹu: float,
    ):
        """
        Reference: (7.45b)
        """
        s = level.s
        l = level.l
        l_u = level_upper.l

        assert s == level_upper.s

        if K != Ku:
            return 0
        if Q != Qu:
            return 0

        transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)
        m0 = (2 * l_u + 1) * transition.einstein_a_ul
        m1 = sqrt((2 * J + 1) * (2 * Jʹ + 1) * (2 * Ju + 1) * (2 * Jʹu + 1))
        m2 = m1p(1 + K + Jʹ + Jʹu)
        m3 = wigner_6j(J, Jʹ, K, Jʹu, Ju, 1)
        m4 = wigner_6j(l_u, l, 1, J, Ju, s)
        m5 = wigner_6j(l_u, l, 1, Jʹ, Jʹu, s)
        result = m0 * m1 * m2 * m3 * m4 * m5
        return result

    def t_e_short(
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

        # rename
        K = k
        Q = q
        J = j
        Jʹ = j_prime
        Ku = k_u
        Qu = q_u
        Ju = j_u
        Jʹu = j_prime_u

        S = level.s
        L = level.l
        Lu = level_upper.l

        transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)

        result = multiply(
            lambda: δ(S, level_upper.s) * δ(K, Ku) * δ(Q, Qu),
            lambda: (2 * Lu + 1) * transition.einstein_a_ul,
            lambda: sqrt(n_proj(J, Jʹ, Ju, Jʹu)),
            lambda: m1p(1 + K + Jʹ + Jʹu),
            lambda: wigner_6j(J, Jʹ, K, Jʹu, Ju, 1),
            lambda: wigner_6j(Lu, L, 1, J, Ju, S),
            lambda: wigner_6j(Lu, L, 1, Jʹ, Jʹu, S),
        )

        return result

    def t_s(
        self,
        level: Level,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        level_upper: Level,
        Ku: int,
        Qu: int,
        Ju: float,
        Jʹu: float,
    ):
        """
        Reference: (7.45c)
        """
        s = level.s
        l = level.l
        l_u = level_upper.l

        assert s == level_upper.s

        transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)

        m0 = (2 * l_u + 1) * transition.einstein_b_ul
        result = 0
        for k_r in [0, 1, 2]:
            for q_r in projection(k_r):  # todo: no need to sum: q_r = q_u - q
                m1 = sqrt(
                    3
                    * (2 * J + 1)
                    * (2 * Jʹ + 1)
                    * (2 * Ju + 1)
                    * (2 * Jʹu + 1)
                    * (2 * K + 1)
                    * (2 * Ku + 1)
                    * (2 * k_r + 1)
                )
                m2 = m1p(k_r + Ku + Qu + Jʹu - Ju)
                m3 = w9j(J, Ju, 1, Jʹ, Jʹu, 1, K, Ku, k_r)
                m4 = wigner_6j(l_u, l, 1, J, Ju, s)
                m5 = wigner_6j(l_u, l, 1, Jʹ, Jʹu, s)
                m6 = wigner_3j(K, Ku, k_r, -Q, Qu, -q_r)
                m7 = self.radiation_tensor.get(transition=transition, k=k_r, q=q_r)
                result += m0 * m1 * m2 * m3 * m4 * m5 * m6 * m7
        return result

    def get_solution_direct(self) -> Rho:
        # A x = 0
        # let x[0] = 1 (it is always rho_0_0 of some kind, so it is unlikely to be exactly zero)
        # A[1:] x = 0
        # A[1:, 1:] x[1:] = -A[1:, 0]
        sol = np.linalg.solve(
            self.matrix_builder.rho_matrix[:, 1:, 1:],
            -self.matrix_builder.rho_matrix[:, 1:, 0:1],
        )
        sol = np.insert(sol, 0, 1.0, 1)
        sol = sol[:, :, 0]

        # Sum sqrt(2J+1) rho00(J, J) = 1
        weights = np.zeros_like(sol)
        for index, weight in zip(self.matrix_builder.trace_indexes, self.matrix_builder.trace_weights):
            weights[:, index] = weight
        trace = (sol * weights).sum(axis=1, keepdims=True)

        solution_vector = sol / trace

        rho = Rho()
        for index, (level_id, k, q, j, j_prime) in self.matrix_builder.index_to_parameters.items():
            rho.set_from_level_id(level_id=level_id, K=k, Q=q, J=j, Jʹ=j_prime, value=solution_vector[:, index])

        return rho
