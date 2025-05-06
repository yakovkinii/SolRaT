import logging

import numpy as np
from numpy import imag, pi, real, sqrt

from src.core.engine.functions.general import m1p, n_proj
from src.core.engine.functions.looping import FROMTO, INTERSECTION, PROJECTION, TRIANGULAR, VALUE
from src.core.engine.generators.multiply import multiply
from src.core.engine.generators.summate import summate
from src.core.physics.constants import c_cm_sm1, h_erg_s, sqrt_pi
from src.core.physics.functions import energy_cmm1_to_frequency_hz
from src.core.physics.rotation_tensor_t_k_q import t_k_q
from src.core.physics.voigt_profile import voigt
from src.core.physics.wigner_3j_6j_9j import wigner_3j, wigner_6j
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.rho_matrix_builder import Rho
from src.two_term_atom.physics.paschen_back import calculate_paschen_back
from src.two_term_atom.terms_levels_transitions.term_registry import Level
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class RadiativeTransferCoefficients:
    def __init__(
        self,
        atmosphere_parameters: AtmosphereParameters,
        transition_registry: TransitionRegistry,
        nu: np.ndarray,
        maximum_delta_v_thermal_units_cutoff=5,
        chi=0,
        theta=0,
        gamma=0,
        N=1,
    ):
        self.atmosphere_parameters: AtmosphereParameters = atmosphere_parameters
        self.transition_registry: TransitionRegistry = transition_registry
        self.nu = nu
        self.maximum_delta_v_thermal_units_cutoff = maximum_delta_v_thermal_units_cutoff
        self.chi = chi
        self.theta = theta
        self.gamma = gamma
        self.N = N  # Atom concentration

    def phi(self, nui, nu):
        """
        Reference: (5.43 - 5.45)
        """
        delta_nu_D = nui * self.atmosphere_parameters.delta_v_thermal_cm_sm1 / c_cm_sm1  # Doppler width

        nu_round = (nui - nu) / delta_nu_D  # nui already accounts for magnetic shifts
        nu_round_A = (
            self.atmosphere_parameters.macroscopic_velocity_cm_sm1 / self.atmosphere_parameters.delta_v_thermal_cm_sm1
        )

        complex_voigt = voigt(nu=nu_round - nu_round_A, a=self.atmosphere_parameters.voigt_a) / sqrt_pi / delta_nu_D
        return complex_voigt

    def cutoff_condition(self, level_upper: Level, level_lower: Level, nu: np.ndarray):
        nui = energy_cmm1_to_frequency_hz(level_upper.get_mean_energy_cmm1() - level_lower.get_mean_energy_cmm1())
        cutoff = (
            self.maximum_delta_v_thermal_units_cutoff
            * nui
            * self.atmosphere_parameters.delta_v_thermal_cm_sm1
            / c_cm_sm1
        )
        if min(nu) > nui + cutoff or max(nu) < nui - cutoff:
            return True
        return False

    def eta_a(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (7.47a)
        """

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

            return summate(
                lambda K, Q, Kl, Ql, jl, Jl, Jʹl, Jʹʹl, ju, Ju, Jʹu, Ml, Mʹl, Mu, q, qʹ: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N * n_proj(Ll),
                    lambda: transition.einstein_b_lu * sqrt(n_proj(1, K, Kl)),
                    lambda: m1p(1 + Jʹʹl - Ml + qʹ),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jl, level=level_lower, M=Ml),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jʹʹl, level=level_lower, M=Ml),
                    lambda: upper_pb_eigenvectors(j=ju, J=Ju, level=level_upper, M=Mu),
                    lambda: upper_pb_eigenvectors(j=ju, J=Jʹu, level=level_upper, M=Mu),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mu, Mʹl, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹʹl, Jʹl, Kl, Ml, -Mʹl, -Ql),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                    lambda: real(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                            lambda: rho(level=level_lower, K=Kl, Q=Ql, J=Jʹʹl, Jʹ=Jʹl),
                            lambda: self.phi(
                                nui=energy_cmm1_to_frequency_hz(
                                    upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu)
                                    - lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
                                ),
                                nu=self.nu,
                            ),
                            is_complex=True,
                        )
                    ),
                ),
                jl=TRIANGULAR(Ll, S),
                Jl=TRIANGULAR(Ll, S),
                Jʹl=TRIANGULAR(Ll, S),
                Jʹʹl=TRIANGULAR(Ll, S),
                ju=TRIANGULAR(Lu, S),
                Ju=INTERSECTION(TRIANGULAR(Lu, S), TRIANGULAR("Jl", 1)),
                Jʹu=INTERSECTION(TRIANGULAR(Lu, S), TRIANGULAR("Jʹl", 1)),
                Ml=INTERSECTION(PROJECTION("Jl"), PROJECTION("Jʹʹl"), PROJECTION("jl")),
                Mʹl=PROJECTION("Jʹl"),
                Mu=INTERSECTION(PROJECTION("Ju"), PROJECTION("Jʹu"), PROJECTION("ju")),
                K=FROMTO(0, 2),
                Kl=TRIANGULAR("Jʹl", "Jʹʹl"),
                Ql=INTERSECTION(PROJECTION("Kl"), VALUE("Ml - Mʹl")),
                q=VALUE("Ml - Mu"),
                qʹ=VALUE("Mʹl - Mu"),
                Q=INTERSECTION(PROJECTION("K"), VALUE("q - qʹ")),
                tqdm_level=1,
            )

    def eta_s(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (7.47b)
        """
        logging.info("Radiative Transfer Equations: calculate eta_s")

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

            return summate(
                lambda ju, Ju, Jʹu, Jʹʹu, jl, Jl, Jʹl, Mu, Mʹu, Ml, K, Q, Ku, Qu, q, qʹ: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N,
                    lambda: n_proj(Lu) * transition.einstein_b_ul * sqrt(n_proj(1, K, Ku)),
                    lambda: m1p(1 + Jʹu - Mu + qʹ),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jl, level=level_lower, M=Ml),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jʹl, level=level_lower, M=Ml),
                    lambda: upper_pb_eigenvectors(j=ju, J=Ju, level=level_upper, M=Mu),
                    lambda: upper_pb_eigenvectors(j=ju, J=Jʹʹu, level=level_upper, M=Mu),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mʹu, Ml, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹu, Jʹʹu, Ku, Mʹu, -Mu, -Qu),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                    lambda: real(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                            lambda: rho(level=level_upper, K=Ku, Q=Qu, J=Jʹu, Jʹ=Jʹʹu),
                            lambda: self.phi(
                                nui=energy_cmm1_to_frequency_hz(
                                    upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu)
                                    - lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
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
                Qu=INTERSECTION(PROJECTION("Ku"), VALUE("Mʹu - Mu")),
                q=VALUE("Ml - Mu"),
                qʹ=VALUE("Ml - Mʹu"),
                Q=INTERSECTION(PROJECTION("K"), VALUE("q - qʹ")),
                tqdm_level=1,
            )

    def rho_a(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (7.47a)
        """

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

            return summate(
                lambda K, Q, Kl, Ql, jl, Jl, Jʹl, Jʹʹl, ju, Ju, Jʹu, Ml, Mʹl, Mu, q, qʹ: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N * n_proj(Ll),
                    lambda: transition.einstein_b_lu * sqrt(n_proj(1, K, Kl)),
                    lambda: m1p(1 + Jʹʹl - Ml + qʹ),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jl, level=level_lower, M=Ml),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jʹʹl, level=level_lower, M=Ml),
                    lambda: upper_pb_eigenvectors(j=ju, J=Ju, level=level_upper, M=Mu),
                    lambda: upper_pb_eigenvectors(j=ju, J=Jʹu, level=level_upper, M=Mu),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mu, Mʹl, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹʹl, Jʹl, Kl, Ml, -Mʹl, -Ql),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                    lambda: imag(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                            lambda: rho(level=level_lower, K=Kl, Q=Ql, J=Jʹʹl, Jʹ=Jʹl),
                            lambda: self.phi(
                                nui=energy_cmm1_to_frequency_hz(
                                    upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu)
                                    - lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
                                ),
                                nu=self.nu,
                            ),
                            is_complex=True,
                        )
                    ),
                ),
                jl=TRIANGULAR(Ll, S),
                Jl=TRIANGULAR(Ll, S),
                Jʹl=TRIANGULAR(Ll, S),
                Jʹʹl=TRIANGULAR(Ll, S),
                ju=TRIANGULAR(Lu, S),
                Ju=INTERSECTION(TRIANGULAR(Lu, S), TRIANGULAR("Jl", 1)),
                Jʹu=INTERSECTION(TRIANGULAR(Lu, S), TRIANGULAR("Jʹl", 1)),
                Ml=INTERSECTION(PROJECTION("Jl"), PROJECTION("Jʹʹl"), PROJECTION("jl")),
                Mʹl=PROJECTION("Jʹl"),
                Mu=INTERSECTION(PROJECTION("Ju"), PROJECTION("Jʹu"), PROJECTION("ju")),
                K=FROMTO(0, 2),
                Kl=TRIANGULAR("Jʹl", "Jʹʹl"),
                Ql=INTERSECTION(PROJECTION("Kl"), VALUE("Ml - Mʹl")),
                q=VALUE("Ml - Mu"),
                qʹ=VALUE("Mʹl - Mu"),
                Q=INTERSECTION(PROJECTION("K"), VALUE("q - qʹ")),
                tqdm_level=1,
            )

    def rho_s(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (7.47b)
        """
        logging.info("Radiative Transfer Equations: calculate eta_s")

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

            return summate(
                lambda ju, Ju, Jʹu, Jʹʹu, jl, Jl, Jʹl, Mu, Mʹu, Ml, K, Q, Ku, Qu, q, qʹ: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N,
                    lambda: n_proj(Lu) * transition.einstein_b_ul * sqrt(n_proj(1, K, Ku)),
                    lambda: m1p(1 + Jʹu - Mu + qʹ),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jl, level=level_lower, M=Ml),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jʹl, level=level_lower, M=Ml),
                    lambda: upper_pb_eigenvectors(j=ju, J=Ju, level=level_upper, M=Mu),
                    lambda: upper_pb_eigenvectors(j=ju, J=Jʹʹu, level=level_upper, M=Mu),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mʹu, Ml, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹu, Jʹʹu, Ku, Mʹu, -Mu, -Qu),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                    lambda: imag(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                            lambda: rho(level=level_upper, K=Ku, Q=Qu, J=Jʹu, Jʹ=Jʹʹu),
                            lambda: self.phi(
                                nui=energy_cmm1_to_frequency_hz(
                                    upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu)
                                    - lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
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
                Qu=INTERSECTION(PROJECTION("Ku"), VALUE("Mʹu - Mu")),
                q=VALUE("Ml - Mu"),
                qʹ=VALUE("Ml - Mʹu"),
                Q=INTERSECTION(PROJECTION("K"), VALUE("q - qʹ")),
                tqdm_level=1,
            )

    @staticmethod
    def epsilon(eta_s: np.ndarray, nu: np.ndarray):
        """
        Reference:
        (7.47e)
        """
        return 2 * h_erg_s * nu**3 / c_cm_sm1**2 * eta_s

    """
    Combined
    Note: 
    1. Returning complex values for eta_rho gives 2x performance benefit.
    2. Likely there is a possibility of calculating multiple Stokes parameters by overloading summate/multiply. 
    """

    def eta_rho_a(self, rho: Rho, stokes_component_index: int):
        """
        eta_a = real(eta_rho_a)
        rho_a = imag(eta_rho_a)
        """

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

            return summate(
                lambda K, Q, Kl, Ql, jl, Jl, Jʹl, Jʹʹl, ju, Ju, Jʹu, Ml, Mʹl, Mu, q, qʹ: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N * n_proj(Ll),
                    lambda: transition.einstein_b_lu * sqrt(n_proj(1, K, Kl)),
                    lambda: m1p(1 + Jʹʹl - Ml + qʹ),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jl, level=level_lower, M=Ml),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jʹʹl, level=level_lower, M=Ml),
                    lambda: upper_pb_eigenvectors(j=ju, J=Ju, level=level_upper, M=Mu),
                    lambda: upper_pb_eigenvectors(j=ju, J=Jʹu, level=level_upper, M=Mu),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mu, Mʹl, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹʹl, Jʹl, Kl, Ml, -Mʹl, -Ql),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                    lambda: multiply(
                        lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                        lambda: rho(level=level_lower, K=Kl, Q=Ql, J=Jʹʹl, Jʹ=Jʹl),
                        lambda: self.phi(
                            nui=energy_cmm1_to_frequency_hz(
                                upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu)
                                - lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
                            ),
                            nu=self.nu,
                        ),
                        is_complex=True,
                    ),
                ),
                jl=TRIANGULAR(Ll, S),
                Jl=TRIANGULAR(Ll, S),
                Jʹl=TRIANGULAR(Ll, S),
                Jʹʹl=TRIANGULAR(Ll, S),
                ju=TRIANGULAR(Lu, S),
                Ju=INTERSECTION(TRIANGULAR(Lu, S), TRIANGULAR("Jl", 1)),
                Jʹu=INTERSECTION(TRIANGULAR(Lu, S), TRIANGULAR("Jʹl", 1)),
                Ml=INTERSECTION(PROJECTION("Jl"), PROJECTION("Jʹʹl"), PROJECTION("jl")),
                Mʹl=PROJECTION("Jʹl"),
                Mu=INTERSECTION(PROJECTION("Ju"), PROJECTION("Jʹu"), PROJECTION("ju")),
                K=FROMTO(0, 2),
                Kl=TRIANGULAR("Jʹl", "Jʹʹl"),
                Ql=INTERSECTION(PROJECTION("Kl"), VALUE("Ml - Mʹl")),
                q=VALUE("Ml - Mu"),
                qʹ=VALUE("Mʹl - Mu"),
                Q=INTERSECTION(PROJECTION("K"), VALUE("q - qʹ")),
                tqdm_level=1,
            )

    def eta_rho_s(self, rho: Rho, stokes_component_index: int):
        """
        eta_s = real(eta_rho_s)
        rho_s = imag(eta_rho_s)
        """
        logging.info("Radiative Transfer Equations: calculate eta_rho_s")

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

            return summate(
                lambda ju, Ju, Jʹu, Jʹʹu, jl, Jl, Jʹl, Mu, Mʹu, Ml, K, Q, Ku, Qu, q, qʹ: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N,
                    lambda: n_proj(Lu) * transition.einstein_b_ul * sqrt(n_proj(1, K, Ku)),
                    lambda: m1p(1 + Jʹu - Mu + qʹ),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jl, level=level_lower, M=Ml),
                    lambda: lower_pb_eigenvectors(j=jl, J=Jʹl, level=level_lower, M=Ml),
                    lambda: upper_pb_eigenvectors(j=ju, J=Ju, level=level_upper, M=Mu),
                    lambda: upper_pb_eigenvectors(j=ju, J=Jʹʹu, level=level_upper, M=Mu),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mʹu, Ml, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹu, Jʹʹu, Ku, Mʹu, -Mu, -Qu),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                    lambda: multiply(
                        lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                        lambda: rho(level=level_upper, K=Ku, Q=Qu, J=Jʹu, Jʹ=Jʹʹu),
                        lambda: self.phi(
                            nui=energy_cmm1_to_frequency_hz(
                                upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu)
                                - lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
                            ),
                            nu=self.nu,
                        ),
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
                Qu=INTERSECTION(PROJECTION("Ku"), VALUE("Mʹu - Mu")),
                q=VALUE("Ml - Mu"),
                qʹ=VALUE("Ml - Mʹu"),
                Q=INTERSECTION(PROJECTION("K"), VALUE("q - qʹ")),
                tqdm_level=1,
            )

    """
    The following are some analytical expressions under further assumptions for validation.
    """

    def eta_a_no_field_no_fine_structure(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (7.48a)
        """

        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower
            Ll = level_lower.L
            Lu = level_upper.L
            S = level_lower.S

            return summate(
                lambda K, Q, Jl, Jʹl: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N * n_proj(Ll),
                    lambda: transition.einstein_b_lu,
                    lambda: m1p(1 - Lu + S + Jʹl),
                    lambda: sqrt(n_proj(1, Jl, Jʹl)),
                    lambda: wigner_6j(Ll, Ll, K, Jl, Jʹl, S),
                    lambda: wigner_6j(1, 1, K, Ll, Ll, Lu),
                    lambda: real(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                            lambda: rho(level=level_lower, K=K, Q=Q, J=Jl, Jʹ=Jʹl),
                            lambda: self.phi(
                                nui=energy_cmm1_to_frequency_hz(
                                    level_upper.get_mean_energy_cmm1() - level_lower.get_mean_energy_cmm1()
                                ),
                                nu=self.nu,
                            ),
                            is_complex=True,
                        )
                    ),
                ),
                Jl=TRIANGULAR(Ll, S),
                Jʹl=TRIANGULAR(Ll, S),
                K=FROMTO(0, 2),
                Q=PROJECTION("K"),
                tqdm_level=1,
            )

    def eta_s_no_field_no_fine_structure(self, rho: Rho, stokes_component_index: int):
        """
        No magnetic field, no fine structure splitting.
        Reference:
        (7.48d)
        """
        logging.info("Radiative Transfer Equations: calculate eta_s")

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

            return summate(
                lambda Ju, Jʹu, K, Q: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N,
                    lambda: n_proj(Lu) * transition.einstein_b_ul,
                    lambda: m1p(1 - Ll + S + Ju + K),
                    lambda: sqrt(n_proj(1, Ju, Jʹu)),
                    lambda: wigner_6j(Lu, Lu, K, Ju, Jʹu, S),
                    lambda: wigner_6j(1, 1, K, Lu, Lu, Ll),
                    lambda: real(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                            lambda: rho(level=level_upper, K=K, Q=Q, J=Jʹu, Jʹ=Ju),
                            lambda: self.phi(
                                nui=energy_cmm1_to_frequency_hz(
                                    level_upper.get_mean_energy_cmm1() - level_lower.get_mean_energy_cmm1()
                                ),
                                nu=self.nu,
                            ),
                        )
                    ),
                ),
                Ju=TRIANGULAR(Lu, S),
                Jʹu=TRIANGULAR(Lu, S),
                K=FROMTO(0, 2),
                Q=INTERSECTION(PROJECTION("K")),
                tqdm_level=1,
            )

    def rho_a_no_field_no_fine_structure(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (7.48a)
        """

        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower
            Ll = level_lower.L
            Lu = level_upper.L
            S = level_lower.S

            return summate(
                lambda K, Q, Jl, Jʹl: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N * n_proj(Ll),
                    lambda: transition.einstein_b_lu,
                    lambda: m1p(1 - Lu + S + Jʹl),
                    lambda: sqrt(n_proj(1, Jl, Jʹl)),
                    lambda: wigner_6j(Ll, Ll, K, Jl, Jʹl, S),
                    lambda: wigner_6j(1, 1, K, Ll, Ll, Lu),
                    lambda: imag(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                            lambda: rho(level=level_lower, K=K, Q=Q, J=Jl, Jʹ=Jʹl),
                            lambda: self.phi(
                                nui=energy_cmm1_to_frequency_hz(
                                    level_upper.get_mean_energy_cmm1() - level_lower.get_mean_energy_cmm1()
                                ),
                                nu=self.nu,
                            ),
                            is_complex=True,
                        )
                    ),
                ),
                Jl=TRIANGULAR(Ll, S),
                Jʹl=TRIANGULAR(Ll, S),
                K=FROMTO(0, 2),
                Q=PROJECTION("K"),
                tqdm_level=1,
            )

    def rho_s_no_field_no_fine_structure(self, rho: Rho, stokes_component_index: int):
        """
        No magnetic field, no fine structure splitting.
        Reference:
        (7.48d)
        """
        logging.info("Radiative Transfer Equations: calculate eta_s")

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

            return summate(
                lambda Ju, Jʹu, K, Q: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N,
                    lambda: n_proj(Lu) * transition.einstein_b_ul,
                    lambda: m1p(1 - Ll + S + Ju + K),
                    lambda: sqrt(n_proj(1, Ju, Jʹu)),
                    lambda: wigner_6j(Lu, Lu, K, Ju, Jʹu, S),
                    lambda: wigner_6j(1, 1, K, Lu, Lu, Ll),
                    lambda: imag(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                            lambda: rho(level=level_upper, K=K, Q=Q, J=Jʹu, Jʹ=Ju),
                            lambda: self.phi(
                                nui=energy_cmm1_to_frequency_hz(
                                    level_upper.get_mean_energy_cmm1() - level_lower.get_mean_energy_cmm1()
                                ),
                                nu=self.nu,
                            ),
                        )
                    ),
                ),
                Ju=TRIANGULAR(Lu, S),
                Jʹu=TRIANGULAR(Lu, S),
                K=FROMTO(0, 2),
                Q=INTERSECTION(PROJECTION("K")),
                tqdm_level=1,
            )

    def eta_s_no_field(self, rho: Rho, stokes_component_index: int):
        """
        No magnetic field.
        Reference:
        (10.127)
        """
        logging.info("Radiative Transfer Equations: calculate eta_s_analytic_resonance")
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

            result = result + summate(
                lambda Ju, Jʹu, Jl, K, Q: multiply(
                    lambda: h_erg_s * self.nu / 4 / pi * self.N * n_proj(Lu) * transition.einstein_b_ul,
                    lambda: m1p(1 + Jl + Jʹu),
                    lambda: sqrt(n_proj(Jl, Jl, 1, Ju, Jʹu)),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Jʹu, S),
                    lambda: wigner_6j(1, 1, K, Ju, Jʹu, Jl),
                    lambda: real(
                        multiply(
                            lambda: t_k_q(K, Q, stokes_component_index, self.chi, self.theta, self.gamma),
                            lambda: rho(level=level_upper, K=K, Q=Q, J=Jʹu, Jʹ=Ju),
                            lambda: 0.5
                            * (
                                self.phi(
                                    nui=energy_cmm1_to_frequency_hz(
                                        level_upper.get_term(J=Ju).energy_cmm1 - level_lower.get_term(J=Jl).energy_cmm1,
                                    ),
                                    nu=self.nu,
                                )
                                + np.conjugate(
                                    self.phi(
                                        nui=energy_cmm1_to_frequency_hz(
                                            level_upper.get_term(J=Jʹu).energy_cmm1
                                            - level_lower.get_term(J=Jl).energy_cmm1,
                                        ),
                                        nu=self.nu,
                                    )
                                )
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
