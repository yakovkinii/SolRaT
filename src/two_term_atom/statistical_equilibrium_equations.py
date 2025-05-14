import logging

import numpy as np
import pandas as pd
from numpy import pi, sqrt
from tqdm import tqdm

from src.core.engine.functions.general import delta, m1p, n_proj
from src.core.engine.functions.looping import FROMTO, INTERSECTION, PROJECTION, TRIANGULAR, VALUE
from src.core.engine.generators.multiply import multiply
from src.core.engine.generators.nested_loops import nested_loops
from src.core.engine.generators.summate import summate
from src.core.physics.functions import energy_cmm1_to_frequency_hz
from src.core.physics.wigner_3j_6j_9j import wigner_3j, wigner_6j, wigner_9j
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.object.rho_matrix_builder import Level, Rho, RhoMatrixBuilder
from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class TwoTermAtom:
    def __init__(
        self,
        term_registry: TermRegistry,
        transition_registry: TransitionRegistry,
        atmosphere_parameters: AtmosphereParameters,
        radiation_tensor: RadiationTensor,
        # n_frequencies: int = 1,
        disable_r_s: bool = False,
        disable_n: bool = False,
    ):
        n_frequencies = 1
        self.term_registry: TermRegistry = term_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.matrix_builder: RhoMatrixBuilder = RhoMatrixBuilder(
            levels=list(self.term_registry.levels.values()), n_frequencies=n_frequencies
        )
        self.atmosphere_parameters: AtmosphereParameters = atmosphere_parameters
        self.radiation_tensor: RadiationTensor = radiation_tensor
        self.disable_r_s = disable_r_s
        self.disable_n = disable_n

    def add_all_equations(self):
        """
        Loops through all equations.

        Reference: (7.38)
        """
        logging.info("Populate Statistical Equilibrium Equations")
        self.matrix_builder.reset_matrix()
        for level in tqdm(self.term_registry.levels.values(), leave=False):
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(level.L, level.S),
                Jʹ=TRIANGULAR(level.L, level.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                self.matrix_builder.select_equation(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.add_coherence_decay(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.add_absorption(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.add_emission(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.add_relaxation(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)

    def add_coherence_decay(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        """
        Reference: (7.38)
        """
        for Jʹʹ, Jʹʹʹ, Kʹ, Qʹ in nested_loops(
            Jʹʹ=INTERSECTION(TRIANGULAR(level.L, level.S), TRIANGULAR(J, 1)),
            Jʹʹʹ=INTERSECTION(TRIANGULAR(level.L, level.S), TRIANGULAR(Jʹ, 1)),
            Kʹ=INTERSECTION(TRIANGULAR(K, 1), TRIANGULAR("Jʹʹ", "Jʹʹʹ")),
            Qʹ=INTERSECTION(PROJECTION("Kʹ"), VALUE(Q)),
        ):
            n = self.n(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            self.matrix_builder.add_coefficient(level=level, K=Kʹ, Q=Qʹ, J=Jʹʹ, Jʹ=Jʹʹʹ, coefficient=-2 * pi * 1j * n)

    def add_absorption(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        """
        Reference: (7.38)
        """
        # Absorption toward selected coherence
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level, level_lower=level_lower):
                continue
            for Jl, Jʹl, Kl, Ql in nested_loops(
                Jl=INTERSECTION(TRIANGULAR(level_lower.L, level_lower.S), TRIANGULAR(J, 1)),
                Jʹl=INTERSECTION(TRIANGULAR(level_lower.L, level_lower.S), TRIANGULAR(Jʹ, 1)),
                Kl=TRIANGULAR("Jl", "Jʹl"),
                Ql=PROJECTION("Kl"),
            ):
                t_a = self.t_a(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, level_lower=level_lower, Kl=Kl, Ql=Ql, Jl=Jl, Jʹl=Jʹl)
                self.matrix_builder.add_coefficient(level=level_lower, K=Kl, Q=Ql, J=Jl, Jʹ=Jʹl, coefficient=t_a)

    def add_emission(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        """
        Reference: (7.38)
        """
        # Emission toward selected coherence
        for level_upper in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level_upper, level_lower=level):
                continue

            for Ju, Jʹu, Ku, Qu in nested_loops(
                Ju=INTERSECTION(TRIANGULAR(level_upper.L, level_upper.S), TRIANGULAR(J, 1)),
                Jʹu=INTERSECTION(TRIANGULAR(level_upper.L, level_upper.S), TRIANGULAR(Jʹ, 1)),
                Ku=TRIANGULAR("Ju", "Jʹu"),
                Qu=PROJECTION("Ku"),
            ):
                t_e = self.t_e(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, level_upper=level_upper, Ku=Ku, Qu=Qu, Ju=Ju, Jʹu=Jʹu)
                t_s = self.t_s(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, level_upper=level_upper, Ku=Ku, Qu=Qu, Ju=Ju, Jʹu=Jʹu)
                self.matrix_builder.add_coefficient(level=level_upper, K=Ku, Q=Qu, J=Ju, Jʹ=Jʹu, coefficient=t_e + t_s)

    def add_relaxation(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        """
        Reference: (7.38)
        """
        # Relaxation from selected coherence
        for Jʹʹ, Jʹʹʹ, Kʹ, Qʹ in nested_loops(
            Jʹʹ=TRIANGULAR(level.L, level.S),
            Jʹʹʹ=TRIANGULAR(level.L, level.S),
            Kʹ=INTERSECTION(TRIANGULAR(J, Jʹ), TRIANGULAR("Jʹʹ", "Jʹʹʹ")),
            Qʹ=PROJECTION("Kʹ"),
        ):
            r_a = self.r_a(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            r_e = self.r_e(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            if self.disable_r_s:
                r_s = 0
            else:
                r_s = self.r_s(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            self.matrix_builder.add_coefficient(level=level, K=Kʹ, Q=Qʹ, J=Jʹʹ, Jʹ=Jʹʹʹ, coefficient=-(r_a + r_e + r_s))

    def r_a(self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float):
        L = level.L
        S = level.S

        result = 0
        for level_upper in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level_upper, level_lower=level):
                continue

            transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)
            Lu = level_upper.L

            result += summate(
                lambda Kr, Qr: multiply(
                    lambda: n_proj(L) * transition.einstein_b_lu,
                    lambda: sqrt(n_proj(1, K, Kʹ, Kr)),
                    lambda: m1p(1 + Lu - S + J + Qʹ),
                    lambda: wigner_6j(L, L, Kr, 1, 1, Lu) * wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr),
                    lambda: 0.5 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr),
                    lambda: delta(J, Jʹʹ),
                    lambda: sqrt(n_proj(Jʹ, Jʹʹʹ)),
                    lambda: wigner_6j(L, L, Kr, Jʹʹʹ, Jʹ, S),
                    lambda: wigner_6j(K, Kʹ, Kr, Jʹʹʹ, Jʹ, J),
                ),
                Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(K, Kʹ)),
                Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qʹ - Q)),
            )

            for Kr, Qr in nested_loops(
                Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(K, Kʹ)),
                Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qʹ - Q)),
            ):
                result += (
                    n_proj(L)
                    * transition.einstein_b_lu
                    * sqrt(n_proj(1, K, Kʹ, Kr))
                    * m1p(1 + Lu - S + J + Qʹ)
                    * wigner_6j(L, L, Kr, 1, 1, Lu)
                    * wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr)
                    * 0.5
                    * self.radiation_tensor(transition=transition, K=Kr, Q=Qr)
                    * delta(Jʹ, Jʹʹʹ)
                    * sqrt(n_proj(J, Jʹʹ))
                    * m1p(Jʹʹ - Jʹ + K + Kʹ + Kr)
                    * wigner_6j(L, L, Kr, Jʹʹ, J, S)
                    * wigner_6j(K, Kʹ, Kr, Jʹʹ, J, Jʹ)
                )

        return result

    def r_e(self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float):
        result = 0
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level, level_lower=level_lower):
                continue
            transition = self.transition_registry.get_transition(level_upper=level, level_lower=level_lower)
            result += delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ) * transition.einstein_a_ul
        return result

    def r_s(self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float):
        # (7.46c)
        L = level.L
        S = level.S
        result = 0
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level, level_lower=level_lower):
                continue
            transition = self.transition_registry.get_transition(level_upper=level, level_lower=level_lower)
            Ll = level_lower.L

            result += summate(
                lambda Kr, Qr: multiply(
                    lambda: n_proj(L) * transition.einstein_b_ul,
                    lambda: sqrt(n_proj(1, K, Kʹ, Kr)),
                    lambda: m1p(1 + Ll - S + J + Kr + Qʹ),
                    lambda: wigner_6j(L, L, Kr, 1, 1, Ll),
                    lambda: wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr),
                    lambda: 0.5 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr),
                    lambda: delta(J, Jʹʹ),
                    lambda: sqrt(n_proj(Jʹ, Jʹʹʹ)),
                    lambda: wigner_6j(L, L, Kr, Jʹʹʹ, Jʹ, S),
                    lambda: wigner_6j(K, Kʹ, Kr, Jʹʹʹ, Jʹ, J),
                ),
                Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(K, Kʹ)),
                Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qʹ - Q)),
            )
            result += summate(
                lambda Kr, Qr: multiply(
                    lambda: n_proj(L) * transition.einstein_b_ul,
                    lambda: sqrt(n_proj(1, K, Kʹ, Kr)),
                    lambda: m1p(1 + Ll - S + J + Kr + Qʹ),
                    lambda: wigner_6j(L, L, Kr, 1, 1, Ll),
                    lambda: wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr),
                    lambda: 0.5 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr),
                    lambda: delta(Jʹ, Jʹʹʹ) * sqrt(n_proj(J, Jʹʹ)),
                    lambda: m1p(Jʹʹ - Jʹ + K + Kʹ + Kr),
                    lambda: wigner_6j(L, L, Kr, Jʹʹ, J, S),
                    lambda: wigner_6j(K, Kʹ, Kr, Jʹʹ, J, Jʹ),
                ),
                Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(K, Kʹ)),
                Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qʹ - Q)),
            )

        return result

    @staticmethod
    def gamma(level: Level, J: float, Jʹ: float):
        """
        (7.42)
        """

        S = level.S
        L = level.L
        result = delta(J, Jʹ) * sqrt(J * (J + 1) * (2 * J + 1))
        result += (
            m1p(1 + L + S + J)
            * sqrt((2 * J + 1) * (2 * Jʹ + 1) * S * (S + 1) * (2 * S + 1))
            * wigner_6j(J, Jʹ, 1, S, S, L)
        )
        return result

    def n(self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float):
        """
        Reference: (7.41)
        """
        if self.disable_n:
            return 0

        term = self.term_registry.get_term(level=level, J=J)
        term_prime = self.term_registry.get_term(level=level, J=Jʹ)
        nu = energy_cmm1_to_frequency_hz(term.energy_cmm1 - term_prime.energy_cmm1)

        result = delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ) * nu

        result += (
            delta(Q, Qʹ)
            * self.atmosphere_parameters.nu_larmor
            * m1p(J + Jʹ - Q)
            * sqrt((2 * K + 1) * (2 * Kʹ + 1))
            * wigner_3j(K, Kʹ, 1, -Q, Q, 0)
            * (
                delta(Jʹ, Jʹʹʹ) * self.gamma(level=level, J=J, Jʹ=Jʹʹ) * wigner_6j(K, Kʹ, 1, Jʹʹ, J, Jʹ)
                + delta(J, Jʹʹ)
                * m1p(K - Kʹ)
                * self.gamma(level=level, J=Jʹʹʹ, Jʹ=Jʹ)
                * wigner_6j(K, Kʹ, 1, Jʹʹʹ, Jʹ, J)
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
        S = level.S
        L = level.L
        Ll = level_lower.L

        transition = self.transition_registry.get_transition(level_upper=level, level_lower=level_lower)

        return summate(
            lambda Kr, Qr: multiply(
                lambda: n_proj(Ll) * transition.einstein_b_lu,
                lambda: sqrt(n_proj(1, J, Jʹ, Jl, Jʹl, K, Kl, Kr)),
                lambda: m1p(Kl + Ql + Jʹl - Jl),
                lambda: wigner_9j(J, Jl, 1, Jʹ, Jʹl, 1, K, Kl, Kr),
                lambda: wigner_6j(L, Ll, 1, Jl, J, S),
                lambda: wigner_6j(L, Ll, 1, Jʹl, Jʹ, S),
                lambda: wigner_3j(K, Kl, Kr, -Q, Ql, -Qr),
                lambda: self.radiation_tensor(transition=transition, K=Kr, Q=Qr),
            ),
            Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(K, Kl)),
            Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Ql - Q)),
        )

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

        S = level.S
        L = level.L
        Lu = level_upper.L

        transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)
        assert S == level_upper.S

        result = multiply(
            lambda: delta(S, level_upper.S) * delta(K, Ku) * delta(Q, Qu),
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
        S = level.S
        L = level.L
        Lu = level_upper.L

        transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)

        return summate(
            lambda Kr, Qr: multiply(
                lambda: n_proj(Lu) * transition.einstein_b_ul,
                lambda: sqrt(n_proj(J, Jʹ, Ju, Jʹu, K, Ku, Kr)),
                lambda: m1p(Kr + Ku + Qu + Jʹu - Ju),
                lambda: wigner_9j(J, Ju, 1, Jʹ, Jʹu, 1, K, Ku, Kr),
                lambda: wigner_6j(Lu, L, 1, J, Ju, S),
                lambda: wigner_6j(Lu, L, 1, Jʹ, Jʹu, S),
                lambda: wigner_3j(K, Ku, Kr, -Q, Qu, -Qr),
                lambda: self.radiation_tensor(transition=transition, K=Kr, Q=Qr),
            ),
            Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(Ku, K)),
            Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qu - Q)),
        )

    def get_solution_direct(self) -> Rho:
        """
        # A x = 0
        # let x[0] = 1 (it is always rho_0_0 of some kind, so it is unlikely to be exactly zero)
        # A[1:] x = 0
        # A[1:, 1:] x[1:] = -A[1:, 0]
        """
        logging.info("Get Solution of Statistical Equilibrium Equations")
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

        rho = Rho(levels=list(self.term_registry.levels.values()))
        for index, (level_id, k, q, j, j_prime) in self.matrix_builder.index_to_parameters.items():
            rho.set_from_level_id(level_id=level_id, K=k, Q=q, J=j, Jʹ=j_prime, value=solution_vector[:, index])

        return rho
