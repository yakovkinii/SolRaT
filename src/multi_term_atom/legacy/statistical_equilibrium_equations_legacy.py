import logging

import numpy as np
from numpy import pi, sqrt
from tqdm import tqdm

from src.engine.functions.general import delta, m1p, n_proj
from src.engine.functions.looping import FROMTO, INTERSECTION, PROJECTION, TRIANGULAR, VALUE
from src.engine.generators.multiply import multiply
from src.engine.generators.nested_loops import nested_loops
from src.engine.generators.summate import summate
from src.common.functions import energy_cmm1_to_frequency_hz
from src.common.wigner_3j_6j_9j import wigner_3j, wigner_6j, wigner_9j
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.rho_matrix_builder import Rho, RhoMatrixBuilder, Term
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class MultiTermAtomSEELegacy:
    """
    This is a legacy SEE implementation.
    It is not vectorized and therefore the new SEE implementation should be preferred for synthesis/inversion.
    This one is kept for reference and testing purposes.
    Also, for one-off calculations this one can actually be faster, as pre-computing time in the vectorized
    implementation is comparable to a couple of runs in this implementation.
    """

    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        disable_r_s: bool = False,
        disable_n: bool = False,
    ):
        self.level_registry: LevelRegistry = level_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.matrix_builder: RhoMatrixBuilder = RhoMatrixBuilder(
            terms=list(self.level_registry.terms.values())
        )
        self.disable_r_s = disable_r_s
        self.disable_n = disable_n

    def add_all_equations(
        self, atmosphere_parameters: AtmosphereParameters, radiation_tensor_in_magnetic_frame: RadiationTensor
    ):
        """
        Loops through all equations.
        Reference: (7.38)
        """
        logging.info("Populate Statistical Equilibrium Equations")
        self.matrix_builder.reset_matrix()
        for term in tqdm(self.level_registry.terms.values(), leave=False):
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                self.matrix_builder.select_equation(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ)
                self.add_coherence_decay(
                    term=term,
                    K=K,
                    Q=Q,
                    J=J,
                    Jʹ=Jʹ,
                    atmosphere_parameters=atmosphere_parameters,
                )
                self.add_absorption(
                    term=term,
                    K=K,
                    Q=Q,
                    J=J,
                    Jʹ=Jʹ,
                    radiation_tensor=radiation_tensor_in_magnetic_frame,
                )
                self.add_emission(
                    term=term,
                    K=K,
                    Q=Q,
                    J=J,
                    Jʹ=Jʹ,
                    radiation_tensor=radiation_tensor_in_magnetic_frame,
                )
                self.add_relaxation(
                    term=term,
                    K=K,
                    Q=Q,
                    J=J,
                    Jʹ=Jʹ,
                    radiation_tensor=radiation_tensor_in_magnetic_frame,
                )

    def add_coherence_decay(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        atmosphere_parameters: AtmosphereParameters,
    ):
        """
        Reference: (7.38)
        """
        for Kʹ, Qʹ, Jʹʹ, Jʹʹʹ in nested_loops(
            Kʹ=TRIANGULAR(K, 1),
            Qʹ=PROJECTION("Kʹ"),
            Jʹʹ=INTERSECTION(TRIANGULAR(term.L, term.S)),
            Jʹʹʹ=INTERSECTION(TRIANGULAR(term.L, term.S)),
        ):
            n = self.n(
                term=term,
                K=K,
                Q=Q,
                J=J,
                Jʹ=Jʹ,
                Kʹ=Kʹ,
                Qʹ=Qʹ,
                Jʹʹ=Jʹʹ,
                Jʹʹʹ=Jʹʹʹ,
                atmosphere_parameters=atmosphere_parameters,
            )
            self.matrix_builder.add_coefficient(term=term, K=Kʹ, Q=Qʹ, J=Jʹʹ, Jʹ=Jʹʹʹ, coefficient=-2 * pi * 1j * n)

    def add_absorption(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        radiation_tensor: RadiationTensor,
    ):
        """
        Reference: (7.38)
        """
        # Absorption toward selected coherence
        for term_lower in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term, term_lower=term_lower):
                continue

            for Jl, Jʹl, Kl, Ql in nested_loops(
                Jl=TRIANGULAR(term_lower.L, term_lower.S),
                Jʹl=TRIANGULAR(term_lower.L, term_lower.S),
                Kl=TRIANGULAR("Jl", "Jʹl"),
                Ql=PROJECTION("Kl"),
            ):
                t_a = self.t_a(
                    term=term,
                    K=K,
                    Q=Q,
                    J=J,
                    Jʹ=Jʹ,
                    term_lower=term_lower,
                    Kl=Kl,
                    Ql=Ql,
                    Jl=Jl,
                    Jʹl=Jʹl,
                    radiation_tensor=radiation_tensor,
                )
                self.matrix_builder.add_coefficient(term=term_lower, K=Kl, Q=Ql, J=Jl, Jʹ=Jʹl, coefficient=t_a)

    def add_emission(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        radiation_tensor: RadiationTensor,
    ):
        """
        Reference: (7.38)
        """
        # Emission toward selected coherence
        for term_upper in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term_upper, term_lower=term):
                continue

            for Ju, Jʹu, Ku, Qu in nested_loops(
                Ju=TRIANGULAR(term_upper.L, term_upper.S),
                Jʹu=TRIANGULAR(term_upper.L, term_upper.S),
                Ku=TRIANGULAR("Ju", "Jʹu"),
                Qu=PROJECTION("Ku"),
            ):
                t_e = self.t_e(
                    term=term,
                    K=K,
                    Q=Q,
                    J=J,
                    Jʹ=Jʹ,
                    term_upper=term_upper,
                    Ku=Ku,
                    Qu=Qu,
                    Ju=Ju,
                    Jʹu=Jʹu,
                )
                t_s = self.t_s(
                    term=term,
                    K=K,
                    Q=Q,
                    J=J,
                    Jʹ=Jʹ,
                    term_upper=term_upper,
                    Ku=Ku,
                    Qu=Qu,
                    Ju=Ju,
                    Jʹu=Jʹu,
                    radiation_tensor=radiation_tensor,
                )
                self.matrix_builder.add_coefficient(term=term_upper, K=Ku, Q=Qu, J=Ju, Jʹ=Jʹu, coefficient=t_e + t_s)

    def add_relaxation(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        radiation_tensor: RadiationTensor,
    ):
        """
        Reference: (7.38)
        """
        # Relaxation from selected coherence
        for Kʹ, Qʹ, Jʹʹ, Jʹʹʹ in nested_loops(
            Kʹ=TRIANGULAR(J, Jʹ),
            Qʹ=PROJECTION("Kʹ"),
            Jʹʹ=TRIANGULAR(term.L, term.S),
            Jʹʹʹ=TRIANGULAR(term.L, term.S),
        ):
            r_a = self.r_a(
                term=term,
                K=K,
                Q=Q,
                J=J,
                Jʹ=Jʹ,
                Kʹ=Kʹ,
                Qʹ=Qʹ,
                Jʹʹ=Jʹʹ,
                Jʹʹʹ=Jʹʹʹ,
                radiation_tensor=radiation_tensor,
            )
            r_e = self.r_e(
                term=term,
                K=K,
                Q=Q,
                J=J,
                Jʹ=Jʹ,
                Kʹ=Kʹ,
                Qʹ=Qʹ,
                Jʹʹ=Jʹʹ,
                Jʹʹʹ=Jʹʹʹ,
            )
            if self.disable_r_s:
                r_s = 0
            else:
                r_s = self.r_s(
                    term=term,
                    K=K,
                    Q=Q,
                    J=J,
                    Jʹ=Jʹ,
                    Kʹ=Kʹ,
                    Qʹ=Qʹ,
                    Jʹʹ=Jʹʹ,
                    Jʹʹʹ=Jʹʹʹ,
                    radiation_tensor=radiation_tensor,
                )
            self.matrix_builder.add_coefficient(term=term, K=Kʹ, Q=Qʹ, J=Jʹʹ, Jʹ=Jʹʹʹ, coefficient=-(r_a + r_e + r_s))

    def r_a(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        Kʹ: int,
        Qʹ: int,
        Jʹʹ: float,
        Jʹʹʹ: float,
        radiation_tensor: RadiationTensor,
    ):
        L = term.L
        S = term.S

        result = 0
        for term_upper in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term_upper, term_lower=term):
                continue

            transition = self.transition_registry.get_transition(term_upper=term_upper, term_lower=term)
            Lu = term_upper.L

            result += summate(
                lambda Kr, Qr: multiply(
                    lambda: n_proj(L) * transition.einstein_b_lu,
                    lambda: sqrt(n_proj(1, K, Kʹ, Kr)),
                    lambda: m1p(1 + Lu - S + J + Qʹ),
                    lambda: wigner_6j(L, L, Kr, 1, 1, Lu) * wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr),
                    lambda: 0.5 * radiation_tensor(transition=transition, K=Kr, Q=Qr),
                    lambda: (
                        multiply(
                            lambda: delta(J, Jʹʹ),
                            lambda: sqrt(n_proj(Jʹ, Jʹʹʹ)),
                            lambda: wigner_6j(L, L, Kr, Jʹʹʹ, Jʹ, S),
                            lambda: wigner_6j(K, Kʹ, Kr, Jʹʹʹ, Jʹ, J),
                        )
                        + multiply(
                            lambda: delta(Jʹ, Jʹʹʹ) * sqrt(n_proj(J, Jʹʹ)),
                            lambda: m1p(Jʹʹ - Jʹ + K + Kʹ + Kr),
                            lambda: wigner_6j(L, L, Kr, Jʹʹ, J, S),
                            lambda: wigner_6j(K, Kʹ, Kr, Jʹʹ, J, Jʹ),
                        )
                    ),
                ),
                Kr=FROMTO(0, 2),
                Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qʹ - Q)),
            )
        return result

    def r_e(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        Kʹ: int,
        Qʹ: int,
        Jʹʹ: float,
        Jʹʹʹ: float,
    ):
        result = 0
        for term_lower in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term, term_lower=term_lower):
                continue
            transition = self.transition_registry.get_transition(term_upper=term, term_lower=term_lower)
            result += delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ) * transition.einstein_a_ul
        return result

    def r_s(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        Kʹ: int,
        Qʹ: int,
        Jʹʹ: float,
        Jʹʹʹ: float,
        radiation_tensor: RadiationTensor,
    ):
        # (7.46c)
        L = term.L
        S = term.S
        result = 0
        for term_lower in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term, term_lower=term_lower):
                continue
            transition = self.transition_registry.get_transition(term_upper=term, term_lower=term_lower)
            Ll = term_lower.L

            result += summate(
                lambda Kr, Qr: multiply(
                    lambda: n_proj(L) * transition.einstein_b_ul,
                    lambda: sqrt(n_proj(1, K, Kʹ, Kr)),
                    lambda: m1p(1 + Ll - S + J + Kr + Qʹ),
                    lambda: wigner_6j(L, L, Kr, 1, 1, Ll),
                    lambda: wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr),
                    lambda: 0.5 * radiation_tensor(transition=transition, K=Kr, Q=Qr),
                    lambda: (
                        multiply(
                            lambda: delta(J, Jʹʹ),
                            lambda: sqrt(n_proj(Jʹ, Jʹʹʹ)),
                            lambda: wigner_6j(L, L, Kr, Jʹʹʹ, Jʹ, S),
                            lambda: wigner_6j(K, Kʹ, Kr, Jʹʹʹ, Jʹ, J),
                        )
                        + multiply(
                            lambda: delta(Jʹ, Jʹʹʹ) * sqrt(n_proj(J, Jʹʹ)),
                            lambda: m1p(Jʹʹ - Jʹ + K + Kʹ + Kr),
                            lambda: wigner_6j(L, L, Kr, Jʹʹ, J, S),
                            lambda: wigner_6j(K, Kʹ, Kr, Jʹʹ, J, Jʹ),
                        )
                    ),
                ),
                Kr=FROMTO(0, 2),
                Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qʹ - Q)),
            )

        return result

    @staticmethod
    def gamma(term: Term, J: float, Jʹ: float):
        """
        (7.42)
        """

        S = term.S
        L = term.L
        result = delta(J, Jʹ) * sqrt(J * (J + 1) * (2 * J + 1))
        result += (
            m1p(1 + L + S + J)
            * sqrt((2 * J + 1) * (2 * Jʹ + 1) * S * (S + 1) * (2 * S + 1))
            * wigner_6j(J, Jʹ, 1, S, S, L)
        )
        return result

    def n(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        Kʹ: int,
        Qʹ: int,
        Jʹʹ: float,
        Jʹʹʹ: float,
        atmosphere_parameters: AtmosphereParameters,
    ):
        """
        Reference: (7.41)
        """
        if self.disable_n:
            return 0

        level = self.level_registry.get_level(term=term, J=J)
        level_prime = self.level_registry.get_level(term=term, J=Jʹ)
        nu = energy_cmm1_to_frequency_hz(level.energy_cmm1 - level_prime.energy_cmm1)

        result = delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ) * nu

        result += (
            delta(Q, Qʹ)
            * atmosphere_parameters.nu_larmor
            * m1p(J + Jʹ - Q)
            * sqrt((2 * K + 1) * (2 * Kʹ + 1))
            * wigner_3j(K, Kʹ, 1, -Q, Q, 0)
            * (
                delta(Jʹ, Jʹʹʹ) * self.gamma(term=term, J=J, Jʹ=Jʹʹ) * wigner_6j(K, Kʹ, 1, Jʹʹ, J, Jʹ)
                + delta(J, Jʹʹ) * m1p(K - Kʹ) * self.gamma(term=term, J=Jʹʹʹ, Jʹ=Jʹ) * wigner_6j(K, Kʹ, 1, Jʹʹʹ, Jʹ, J)
            )
        )
        return result

    def t_a(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        term_lower: Term,
        Kl: int,
        Ql: int,
        Jl: float,
        Jʹl: float,
        radiation_tensor: RadiationTensor,
    ):
        """
        Reference: (7.45a)
        """
        S = term.S
        L = term.L
        Ll = term_lower.L

        transition = self.transition_registry.get_transition(term_upper=term, term_lower=term_lower)

        return summate(
            lambda Kr, Qr: multiply(
                lambda: n_proj(Ll) * transition.einstein_b_lu,
                lambda: sqrt(n_proj(1, J, Jʹ, Jl, Jʹl, K, Kl, Kr)),
                lambda: m1p(Kl + Ql + Jʹl - Jl),
                lambda: wigner_9j(J, Jl, 1, Jʹ, Jʹl, 1, K, Kl, Kr),
                lambda: wigner_6j(L, Ll, 1, Jl, J, S),
                lambda: wigner_6j(L, Ll, 1, Jʹl, Jʹ, S),
                lambda: wigner_3j(K, Kl, Kr, -Q, Ql, -Qr),
                lambda: radiation_tensor(transition=transition, K=Kr, Q=Qr),
            ),
            Kr=FROMTO(0, 2),
            Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Ql - Q)),
        )

    def t_e(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        term_upper: Term,
        Ku: int,
        Qu: int,
        Ju: float,
        Jʹu: float,
    ):
        """
        Reference: (7.45b)
        """

        S = term.S
        L = term.L
        Lu = term_upper.L

        transition = self.transition_registry.get_transition(term_upper=term_upper, term_lower=term)
        assert S == term_upper.S

        result = multiply(
            lambda: delta(S, term_upper.S) * delta(K, Ku) * delta(Q, Qu),
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
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        term_upper: Term,
        Ku: int,
        Qu: int,
        Ju: float,
        Jʹu: float,
        radiation_tensor: RadiationTensor,
    ):
        """
        Reference: (7.45c)
        """
        S = term.S
        L = term.L
        Lu = term_upper.L

        transition = self.transition_registry.get_transition(term_upper=term_upper, term_lower=term)

        return summate(
            lambda Kr, Qr: multiply(
                lambda: n_proj(Lu) * transition.einstein_b_ul,
                lambda: sqrt(n_proj(J, Jʹ, Ju, Jʹu, K, Ku, Kr)),
                lambda: m1p(Kr + Ku + Qu + Jʹu - Ju),
                lambda: wigner_9j(J, Ju, 1, Jʹ, Jʹu, 1, K, Ku, Kr),
                lambda: wigner_6j(Lu, L, 1, J, Ju, S),
                lambda: wigner_6j(Lu, L, 1, Jʹ, Jʹu, S),
                lambda: wigner_3j(K, Ku, Kr, -Q, Qu, -Qr),
                lambda: radiation_tensor(transition=transition, K=Kr, Q=Qr),
            ),
            Kr=FROMTO(0, 2),
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

        rho = Rho(terms=list(self.level_registry.terms.values()))
        for index, (term_id, k, q, j, j_prime) in self.matrix_builder.index_to_parameters.items():
            rho.set_from_term_id(term_id=term_id, K=k, Q=q, J=j, Jʹ=j_prime, value=solution_vector[:, index])

        return rho
