import logging

import numpy as np
from numpy import pi, real, sqrt

from src.core.engine.functions.general import m1p, n_proj
from src.core.engine.functions.looping import FROMTO, INTERSECTION, PROJECTION, TRIANGULAR
from src.core.engine.generators.multiply import multiply
from src.core.engine.generators.summate import summate
from src.core.physics.constants import c, h
from src.core.physics.functions import energy_cmm1_to_frequency_hz
from src.core.physics.rotation_tensor_t_k_q import t_k_q
from src.core.physics.wigner_3j_6j_9j import wigner_3j, wigner_6j
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.rho_matrix_builder import Rho
from src.two_term_atom.physics.paschen_back import calculate_paschen_back
from src.two_term_atom.terms_levels_transitions.term_registry import Level, get_transition_frequency
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class RadiativeTransferCoefficients:
    def __init__(
        self,
        atmosphere_parameters: AtmosphereParameters,
        transition_registry: TransitionRegistry,
        nu: np.ndarray,
        maximum_delta_v_thermal_units_cutoff=5,
    ):
        self.atmosphere_parameters: AtmosphereParameters = atmosphere_parameters
        self.transition_registry: TransitionRegistry = transition_registry
        self.nu = nu
        self.maximum_delta_v_thermal_units_cutoff = maximum_delta_v_thermal_units_cutoff

    def phi(self, nui, nu):  # Implement properly
        delta_nu = nui * self.atmosphere_parameters.delta_v_thermal_cm_sm1 / c
        return np.exp(-(((nu - nui) / delta_nu) ** 2))

    def eta_a(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (7.47a)
        """
        chi = 0  # Todo
        theta = 0  # Todo
        gamma = 0  # Todo

        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower
            Ll = level_lower.L
            Lu = level_upper.L
            S = level_lower.S
            lower_pb_eigenvalues, lower_pb_eigenvectors = calculate_paschen_back(
                level=level_lower, magnetic_field_gauss=self.atmosphere_parameters.magnetic_field_gauss
            )
            upper_pb_eigenvalues, upper_pb_eigenvectors = calculate_paschen_back(
                level=level_upper, magnetic_field_gauss=self.atmosphere_parameters.magnetic_field_gauss
            )

            N = 1  # Todo
            return summate(
                lambda K, Q, Kl, Ql, jl, Jl, Jʹl, Jʹʹl, ju, Ju, Jʹu, Ml, Mʹl, Mu, q, qʹ: multiply(
                    lambda: self.nu * h / 4 / pi * N * n_proj(Ll),
                    lambda: transition.einstein_b_lu * sqrt(3 * n_proj(K, Kl)),
                    lambda: m1p(1 + Jʹʹl - Ml + qʹ),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mu, Mʹl, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹʹl, Jʹl, Kl, Ml, -Mʹl, -Ql),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jl, level=level_lower, M=Ml),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jʹʹl, level=level_lower, M=Ml),
                    lambda: upper_pb_eigenvectors(j=ju, J=Ju, level=level_upper, M=Mu),
                    lambda: upper_pb_eigenvectors(j=ju, J=Jʹu, level=level_upper, M=Mu),
                    lambda: real(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, chi, theta, gamma),
                            lambda: rho(level=level_lower, K=Kl, Q=Ql, J=Jʹʹl, Jʹ=Jʹl),
                            lambda: self.phi(
                                nui=get_transition_frequency(
                                    energy_lower_cmm1=lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
                                    energy_upper_cmm1=upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu),
                                ),
                                nu=self.nu,
                            ),
                            is_complex=True,
                        )
                    ),
                ),
                Jl=TRIANGULAR(Ll, S),
                Ml=PROJECTION("Jl"),
                Jʹl=TRIANGULAR(Ll, S),
                Mʹl=PROJECTION("Jʹl"),
                Jʹʹl=TRIANGULAR(Ll, S),
                jl=INTERSECTION(TRIANGULAR(Ll, S)),
                ju=TRIANGULAR(Lu, S),
                Ju=TRIANGULAR(Lu, S),
                Jʹu=TRIANGULAR(Lu, S),
                Mu=PROJECTION("Ju"),
                K=TRIANGULAR(0, 2),
                Q=PROJECTION("K"),
                Kl=TRIANGULAR("Jʹl", "Jʹʹl"),
                Ql=PROJECTION("Kl"),
                q=FROMTO(-1, 1),
                qʹ=FROMTO(-1, 1),
            )

    def cutoff_condition(self, level_upper: Level, level_lower: Level, nu: np.ndarray):
        nui = energy_cmm1_to_frequency_hz(level_upper.get_mean_energy_cmm1() - level_lower.get_mean_energy_cmm1())
        cutoff = self.maximum_delta_v_thermal_units_cutoff * nui * self.atmosphere_parameters.delta_v_thermal_cm_sm1 / c
        if min(nu) > nui + cutoff or max(nu) < nui - cutoff:
            return True
        return False

    def eta_s(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (7.47b)
        """
        logging.info("Radiative Transfer Equations: calculate eta_s")
        chi = 0  # Todo
        theta = 0  # Todo
        gamma = 0  # Todo

        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower

            if self.cutoff_condition(level_upper=level_upper, level_lower=level_lower, nu=self.nu):
                logging.info(
                    f"Cutting off the transition {level_upper.level_id} -> {level_lower.level_id} "
                    f"because it does not contribute to the specified frequency range"
                )
                continue

            logging.info(f"Processing {level_upper.level_id} -> {level_lower.level_id}")

            Ll = level_lower.L
            Lu = level_upper.L
            if abs(Lu - Ll) > 1:
                logging.info(f"Cutting off the transition because |Lu-Ll| > 1")
                continue

            S = level_lower.S
            lower_pb_eigenvalues, lower_pb_eigenvectors = calculate_paschen_back(
                level=level_lower, magnetic_field_gauss=self.atmosphere_parameters.magnetic_field_gauss
            )
            upper_pb_eigenvalues, upper_pb_eigenvectors = calculate_paschen_back(
                level=level_upper, magnetic_field_gauss=self.atmosphere_parameters.magnetic_field_gauss
            )

            N = 1  # Todo

            return summate(
                lambda ju, Ju, Jʹu, Jʹʹu, jl, Jl, Jʹl, Mu, Mʹu, Ml, K, Q, Ku, Qu, q, qʹ: multiply(
                    lambda: h * self.nu / 4 / pi * N * n_proj(Lu) * transition.einstein_b_ul * sqrt(3 * n_proj(K, Ku)),
                    lambda: m1p(1 + Jʹʹu - Mu + qʹ),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mʹu, Ml, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹu, Jʹʹu, Ku, Mʹu, -Mu, -Qu),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jl, level=level_lower, M=Ml),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jʹl, level=level_lower, M=Ml),
                    lambda: upper_pb_eigenvectors(j=ju, J=Ju, level=level_upper, M=Mu),
                    lambda: upper_pb_eigenvectors(j=ju, J=Jʹʹu, level=level_upper, M=Mu),
                    lambda: real(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, chi, theta, gamma),
                            lambda: rho(level=level_upper, K=Ku, Q=Qu, J=Jʹu, Jʹ=Jʹʹu),
                            lambda: self.phi(
                                nui=get_transition_frequency(
                                    energy_lower_cmm1=lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
                                    energy_upper_cmm1=upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu),
                                ),
                                nu=self.nu,
                            ),
                        )
                    ),
                ),
                ju=TRIANGULAR(Lu, S),
                Ju=TRIANGULAR(Lu, S),
                Jʹu=TRIANGULAR(Lu, S),
                Jʹʹu=TRIANGULAR(Lu, S),
                jl=TRIANGULAR(Ll, S),
                Jl=INTERSECTION(TRIANGULAR(Ll, S), TRIANGULAR("Ju", 1)),
                Jʹl=INTERSECTION(TRIANGULAR(Ll, S), TRIANGULAR("Jʹu", 1)),
                Mu=INTERSECTION(PROJECTION("Ju"), PROJECTION("Jʹʹu"), PROJECTION("ju")),
                Mʹu=PROJECTION("Jʹu"),
                Ml=INTERSECTION(PROJECTION("Jl"), PROJECTION("Jʹl"), PROJECTION("jl")),
                K=FROMTO(0, 2),
                Ku=TRIANGULAR("Jʹu", "Jʹʹu"),
                Qu=INTERSECTION(PROJECTION("Ku"), "[Mʹu -Mu]"),
                q="[Ml-Mu]",
                qʹ="[Ml-Mʹu]",
                Q=INTERSECTION(PROJECTION("K"), "[q-qʹ]"),
                tqdm_level=1,
            )

    def eta_s_analytic_resonance(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (10.127)
        """
        logging.info("Radiative Transfer Equations: calculate eta_s_analytic_resonance")
        chi = 0  # Todo
        theta = 0  # Todo
        gamma = 0  # Todo
        result = 0
        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower
            logging.info(f"{level_upper.level_id} -> {level_lower.level_id}")
            if self.cutoff_condition(level_upper=level_upper, level_lower=level_lower, nu=self.nu):
                logging.info(f"Cutting off the transition because it is out of frequency range")
                continue

            Ll = level_lower.L
            Lu = level_upper.L
            S = level_lower.S
            N = 1  # Todo

            result = result + summate(
                lambda Ju, Jʹu, Jl, K, Q: multiply(
                    lambda: h * self.nu / 4 / pi * N * n_proj(Lu) * transition.einstein_b_ul,
                    lambda: m1p(1 + Jʹu + Jl),
                    lambda: sqrt(n_proj(Jl, Jl, 1, Ju, Jʹu)),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Jʹu, S),
                    lambda: wigner_6j(1, 1, K, Ju, Jʹu, Jl),
                    lambda: real(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, chi, theta, gamma),
                            lambda: rho(level=level_upper, K=K, Q=Q, J=Jʹu, Jʹ=Ju),
                            lambda: self.phi(
                                nui=get_transition_frequency(
                                    energy_lower_cmm1=level_lower.get_term(J=Jl).energy_cmm1,
                                    energy_upper_cmm1=level_upper.get_term(J=Ju).energy_cmm1,
                                ),
                                nu=self.nu,
                            ),
                        )
                    ),
                ),
                Ju=TRIANGULAR(Lu, S),
                Jʹu=TRIANGULAR(Lu, S),
                Jl=TRIANGULAR(Ll, S),
                K=FROMTO(0, 2),
                Q=PROJECTION("K"),
            )
        return result
