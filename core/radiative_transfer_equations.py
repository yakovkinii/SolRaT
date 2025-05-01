import logging

import numpy as np
import pandas as pd
from numpy import pi, real, sqrt
from tqdm import tqdm

from core.base.generator import multiply, n_proj, summate, nested_loops
from core.base.math import m1p
from core.base.python import FROMTO, PROJECTION, TRIANGULAR
from core.matrix_builder import Rho
from core.object.atmosphere_parameters import AtmosphereParameters
from core.object.t_k_q import calculate_T_K_Q
from core.steps.paschen_back import calculate_paschen_back
from core.terms_levels_transitions.term_registry import get_transition_frequency
from core.terms_levels_transitions.transition_registry import TransitionRegistry
from core.utility.constant import h, c
from core.utility.wigner_3j_6j_9j import wigner_3j, wigner_6j, check_wigner_3j, check_wigner_6j


class RadiativeTransferCoefficients:
    def __init__(
        self, atmosphere_parameters: AtmosphereParameters, transition_registry: TransitionRegistry, nu: np.ndarray
    ):
        self.atmosphere_parameters: AtmosphereParameters = atmosphere_parameters
        self.transition_registry: TransitionRegistry = transition_registry
        self.nu = nu

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
            Ll = level_lower.l
            Lu = level_upper.l
            S = level_lower.s
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
                            lambda: calculate_T_K_Q(K, Q, stokes_component_index, chi, theta, gamma),
                            lambda: rho(level=level_lower, K=Kl, Q=Ql, J=Jʹʹl, Jʹ=Jʹl),
                            lambda: self.phi(
                                nui=get_transition_frequency(
                                    energy_lower_cmm1=lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
                                    energy_upper_cmm1=upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu),
                                ),
                                nu=self.nu,
                            ),
                            complex=True,
                        )
                    ),
                ),
                jl=TRIANGULAR(Ll, S),
                Jl=TRIANGULAR(Ll, S),
                Jʹl=TRIANGULAR(Ll, S),
                Jʹʹl=TRIANGULAR(Ll, S),
                ju=TRIANGULAR(Lu, S),
                Ju=TRIANGULAR(Lu, S),
                Jʹu=TRIANGULAR(Lu, S),
                Ml=PROJECTION("Jl"),
                Mʹl=PROJECTION("Jʹl"),
                Mu=PROJECTION("Ju"),
                K=TRIANGULAR(0, 2),
                Q=PROJECTION("K"),
                Kl=TRIANGULAR("Jʹl", "Jʹʹl"),
                Ql=PROJECTION("Kl"),
                q=FROMTO(-1, 1),
                qʹ=FROMTO(-1, 1),
            )

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
            Ll = level_lower.l
            Lu = level_upper.l
            S = level_lower.s
            lower_pb_eigenvalues, lower_pb_eigenvectors = calculate_paschen_back(
                level=level_lower, magnetic_field_gauss=self.atmosphere_parameters.magnetic_field_gauss
            )
            upper_pb_eigenvalues, upper_pb_eigenvectors = calculate_paschen_back(
                level=level_upper, magnetic_field_gauss=self.atmosphere_parameters.magnetic_field_gauss
            )

            N = 1  # Todo

            return summate(
                lambda ju, Ju, Jʹu, Jʹʹu, jl, Jl, Jʹl, Mu, Mʹu, Ml, K, Q, Ku, Qu, q, qʹ: multiply(
                    lambda: check_wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: check_wigner_3j(Jʹu, Jʹl, 1, -Mʹu, Ml, -qʹ),
                    lambda: check_wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: check_wigner_3j(Jʹu, Jʹʹu, Ku, Mʹu, -Mu, -Qu),
                    lambda: check_wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: check_wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
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
                            lambda: calculate_T_K_Q(K, Q, stokes_component_index, chi, theta, gamma),
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
                Jl=TRIANGULAR(Ll, S),
                Jʹl=TRIANGULAR(Ll, S),
                Mu=PROJECTION("Ju"),
                Mʹu=PROJECTION("Jʹu"),
                Ml=PROJECTION("Jl"),
                K=FROMTO(0, 2),
                Q=PROJECTION("K"),
                Ku=TRIANGULAR("Jʹu", "Jʹʹu"),
                Qu=PROJECTION("Ku"),
                q=FROMTO(-1, 1),
                qʹ=FROMTO(-1, 1),
            )

    def eta_s_analytic_resonance(self, rho: Rho, stokes_component_index: int):
        """
        Reference:
        (10.127)
        """
        chi = 0  # Todo
        theta = 0  # Todo
        gamma = 0  # Todo

        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower
            Ll = level_lower.l
            Lu = level_upper.l
            S = level_lower.s

            N = 1  # Todo

            return summate(
                lambda Ju, Jʹu, Jl, K, Q: multiply(
                    lambda: h * self.nu / 4 / pi * N * n_proj(Lu) * transition.einstein_b_ul,
                    lambda: m1p(1 + Jʹu + Jl),
                    lambda: sqrt(n_proj(Jl, Jl, 1, Ju, Jʹu)),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Jʹu, S),
                    lambda: wigner_6j(1, 1, K, Ju, Jʹu, Jl),
                    lambda: real(
                        multiply(
                            lambda: calculate_T_K_Q(K, Q, stokes_component_index, chi, theta, gamma),
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
