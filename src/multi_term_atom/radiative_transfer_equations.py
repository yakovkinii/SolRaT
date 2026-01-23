import logging

import numpy as np
import pandas as pd
from numpy import pi, sqrt

from src.common.constants import c_cm_sm1, h_erg_s, sqrt_pi
from src.common.functions import energy_cmm1_to_frequency_hz
from src.common.rotations import T_K_Q_double_rotation, WignerD
from src.common.voigt_profile import voigt
from src.common.wigner_3j_6j_9j import wigner_3j, wigner_6j
from src.engine.functions.decorators import log_method
from src.engine.functions.general import m1p, n_proj
from src.engine.generators.merge_frame import Frame, SumLimits
from src.engine.generators.merge_loopers import (
    ApplyConstraint,
    Constraint,
    DummyOrAlreadyMerged,
    FromTo,
    Intersection,
    Projection,
    Triangular,
)
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiative_transfer_coefficients import (
    RadiativeTransferCoefficients,
)
from src.multi_term_atom.object.rho_matrix_builder import Rho
from src.multi_term_atom.physics.paschen_back import calculate_paschen_back
from src.multi_term_atom.terms_levels_transitions.level_registry import (
    LevelRegistry,
    Term,
)
from src.multi_term_atom.terms_levels_transitions.transition_registry import (
    TransitionRegistry,
)


class MultiTermAtomRTE:
    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        nu: np.ndarray,
        delta_nu_cutoff=None,
        angles: Angles = None,
        magnetic_field_gauss=None,
        rho: Rho = None,
        N: float = 1.0,
        precompute=False,
        j_constrained=False,
    ):
        """
        Radiative Transfer Coefficients within Multi-Term atom model.

        Reference: (7.47)

        :param level_registry:  LevelRegistry instance for the multi-term atom under study.
        :param transition_registry:  TransitionRegistry instance for the multi-term atom under study.
        :param nu:  frequencies [Hz]
        :param delta_nu_cutoff:  distance in frequency for cutting off irrelevant transitions.
        Leave None for a conservative default value.
        :param angles:  Not supported for now.
        :param magnetic_field_gauss:  Not suported for now.
        :param rho:  Not supported for now.
        :param N:  atom numeric concentration for real space (as opposed to density) modeling.
        :param precompute:  Not supported for now.
        :param j_constrained:  constrain J values to the ones specified in transition_registry.
        This parameter is useful for modeling lines like Fe5434 where fine structure components are scattered
        over a very broad spectral interval, while the user is interested only in a specific transition.
        """
        not_supported_message = (
            "Precomputing in RTE has been disabled. The precomputing-related inputs are left here to keep the signature"
            " intact for future development. Please provide these parameters at RTE evaluation step instead."
        )
        assert angles is None, not_supported_message
        assert magnetic_field_gauss is None, not_supported_message
        assert rho is None, not_supported_message
        assert precompute is False, not_supported_message

        self.level_registry: LevelRegistry = level_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.nu = nu
        self.delta_nu_cutoff = delta_nu_cutoff
        if self.delta_nu_cutoff is None:
            self.delta_nu_cutoff = max(10 * (np.max(nu) - np.min(nu)), np.mean(nu) * 1e-3)
        self.N = N
        self.j_constrained = j_constrained

    @log_method
    def calculate_eta_rho_a(
        self,
        stokes_component_index: int,
        angles: Angles,
        rho: Rho,
        atmosphere_parameters: AtmosphereParameters,
    ) -> np.ndarray:
        """
        Calculate etaA and rhoA for selected Stokes component

        Reference: (7.47 ac)

        :param stokes_component_index:  denotes Stokes component (0=I 1=Q 2=U 3=V)
        :param angles:  Angles instance with LOS and magnetic field angles
        :param rho:  density tensor Rho
        :param atmosphere_parameters:  AtmosphereParameters instance
        :return: complex array of etaA + i * rhoA vs frequency
        """
        sum_limits = self.AFrameSumLimitsConstrained() if self.j_constrained else self.AFrameSumLimits()

        frame = Frame.from_sum_limits(
            base_frame=self.create_base_frame(),
            sum_limits=sum_limits,
        )

        # Preparation:
        D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
        D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)
        precalculated_pb_eigenvalues = {}
        precalculated_pb_eigenvectors = {}

        for term in self.level_registry.terms.values():
            pb_eigenvalues, pb_eigenvectors = calculate_paschen_back(
                term=term, magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss
            )
            precalculated_pb_eigenvalues[term.term_id] = pb_eigenvalues
            precalculated_pb_eigenvectors[term.term_id] = pb_eigenvectors

        frame.register_multiplication(
            lambda Ll:                          n_proj(Ll),
            lambda einstein_b_lu, K, Kl:        einstein_b_lu * sqrt(n_proj(1, K, Kl)),
            lambda Jʹʹl, Ml, qʹ:                m1p(1 + Jʹʹl - Ml + qʹ),
            lambda Jl, Jʹl, Ju, Jʹu:            sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
            lambda Ju, Jl, Mu, Ml, q:           wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
            lambda Jʹu, Jʹl, Mu, Mʹl, qʹ:       wigner_3j(Jʹu, Jʹl, 1, -Mu, Mʹl, -qʹ),
            lambda K, q, qʹ, Q:                 wigner_3j(1, 1, K, q, -qʹ, -Q),
            lambda Jʹʹl, Jʹl, Kl, Ml, Mʹl, Ql:  wigner_3j(Jʹʹl, Jʹl, Kl, Ml, -Mʹl, -Ql),
            lambda Lu, Ll, Jl, Ju, S:           wigner_6j(Lu, Ll, 1, Jl, Ju, S),
            lambda Lu, Ll, Jʹl, Jʹu, S:         wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
        )  # fmt: skip

        frame.register_multiplication(
            lambda K, Q: T_K_Q_double_rotation(
                K=K,
                Q=Q,
                stokes_component_index=stokes_component_index,
                D_inverse_omega=D_inverse_omega,
                D_magnetic=D_magnetic,
            ),
            lambda term_lower_id, jl, Jl, Ml: precalculated_pb_eigenvectors[term_lower_id](j=jl, J=Jl, M=Ml),
            lambda term_lower_id, jl, Jʹʹl, Ml: precalculated_pb_eigenvectors[term_lower_id](j=jl, J=Jʹʹl, M=Ml),
            lambda term_upper_id, ju, Ju, Mu: precalculated_pb_eigenvectors[term_upper_id](j=ju, J=Ju, M=Mu),
            lambda term_upper_id, ju, Jʹu, Mu: precalculated_pb_eigenvectors[term_upper_id](j=ju, J=Jʹu, M=Mu),
            lambda term_lower_id, Kl, Ql, Jʹʹl, Jʹl: rho(term_id=term_lower_id, K=Kl, Q=Ql, J=Jʹʹl, Jʹ=Jʹl),
            lambda ju, Mu, term_upper_id, jl, Ml, term_lower_id: self.phi(
                nui=energy_cmm1_to_frequency_hz(
                    precalculated_pb_eigenvalues[term_upper_id](j=ju, M=Mu)
                    - precalculated_pb_eigenvalues[term_lower_id](j=jl, M=Ml)
                ),
                nu=self.nu,
                macroscopic_velocity_cm_sm1=atmosphere_parameters.macroscopic_velocity_cm_sm1,
                delta_v_thermal_cm_sm1=atmosphere_parameters.delta_v_thermal_cm_sm1,
                voigt_a=atmosphere_parameters.voigt_a,
            ),
            elementwise=True,
        )

        result = frame.reduce(
            sum_limits.K,
            sum_limits.Q,
            sum_limits.Ju,
            sum_limits.Jʹu,
            sum_limits.Jl,
            sum_limits.Kl,
            sum_limits.Ql,
            sum_limits.Jʹʹl,
            sum_limits.Jʹl,
            ...,  # Ellipsis means reduce all remaining indexes
        )
        result = h_erg_s * self.nu / 4 / pi * self.N * result
        return result

    @log_method
    def calculate_eta_rho_s(
        self,
        stokes_component_index: int,
        angles: Angles,
        rho: Rho,
        atmosphere_parameters: AtmosphereParameters,
    ):
        """
        Calculate etaS and rhoS for selected Stokes component

        Reference: (7.47 bd)

        :param stokes_component_index:  denotes Stokes component (0=I 1=Q 2=U 3=V)
        :param angles:  Angles instance with LOS and magnetic field angles
        :param rho:  density tensor Rho
        :param atmosphere_parameters:  AtmosphereParameters instance
        :return: complex array of etaS + i * rhoS vs frequency
        """
        sum_limits = self.SFrameSumLimitsConstrained() if self.j_constrained else self.SFrameSumLimits()

        frame = Frame.from_sum_limits(
            base_frame=self.create_base_frame(),
            sum_limits=sum_limits,
        )

        # Preparation:
        D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
        D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)
        precalculated_pb_eigenvalues = {}
        precalculated_pb_eigenvectors = {}
        for term in self.level_registry.terms.values():
            pb_eigenvalues, pb_eigenvectors = calculate_paschen_back(
                term=term, magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss
            )
            precalculated_pb_eigenvalues[term.term_id] = pb_eigenvalues
            precalculated_pb_eigenvectors[term.term_id] = pb_eigenvectors

        frame.register_multiplication(
            lambda Lu:                          n_proj(Lu),
            lambda einstein_b_ul, K, Ku:        einstein_b_ul * sqrt(n_proj(1, K, Ku)),
            lambda Jʹu, Mu, qʹ:                 m1p(1 + Jʹu - Mu + qʹ),
            lambda Jl, Jʹl, Ju, Jʹu:            sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
            lambda Ju, Jl, Mu, Ml, q:           wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
            lambda Jʹu, Jʹl, Mʹu, Ml, qʹ:       wigner_3j(Jʹu, Jʹl, 1, -Mʹu, Ml, -qʹ),
            lambda K, q, qʹ, Q: wigner_3j(1, 1, K, q, -qʹ, -Q),
            lambda Jʹʹu, Jʹu, Ku, Mu, Mʹu, Qu:  wigner_3j(Jʹu, Jʹʹu, Ku, Mʹu, -Mu, -Qu),
            lambda Lu, Ll, Jl, Ju, S:           wigner_6j(Lu, Ll, 1, Jl, Ju, S),
            lambda Lu, Ll, Jʹl, Jʹu, S:         wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
        )  # fmt: skip

        frame.register_multiplication(
            lambda K, Q: T_K_Q_double_rotation(
                K=K,
                Q=Q,
                stokes_component_index=stokes_component_index,
                D_inverse_omega=D_inverse_omega,
                D_magnetic=D_magnetic,
            ),
            lambda term_lower_id, jl, Jl, Ml: precalculated_pb_eigenvectors[term_lower_id](j=jl, J=Jl, M=Ml),
            lambda term_lower_id, jl, Jʹl, Ml: precalculated_pb_eigenvectors[term_lower_id](j=jl, J=Jʹl, M=Ml),
            lambda term_upper_id, ju, Ju, Mu: precalculated_pb_eigenvectors[term_upper_id](j=ju, J=Ju, M=Mu),
            lambda term_upper_id, ju, Jʹʹu, Mu: precalculated_pb_eigenvectors[term_upper_id](j=ju, J=Jʹʹu, M=Mu),
            lambda term_upper_id, Ku, Qu, Jʹʹu, Jʹu: rho(term_id=term_upper_id, K=Ku, Q=Qu, J=Jʹu, Jʹ=Jʹʹu),
            lambda ju, Mu, term_upper_id, jl, Ml, term_lower_id: self.phi(
                nui=energy_cmm1_to_frequency_hz(
                    precalculated_pb_eigenvalues[term_upper_id](j=ju, M=Mu)
                    - precalculated_pb_eigenvalues[term_lower_id](j=jl, M=Ml)
                ),
                nu=self.nu,
                macroscopic_velocity_cm_sm1=atmosphere_parameters.macroscopic_velocity_cm_sm1,
                delta_v_thermal_cm_sm1=atmosphere_parameters.delta_v_thermal_cm_sm1,
                voigt_a=atmosphere_parameters.voigt_a,
            ),
            elementwise=True,
        )

        result = frame.reduce(
            sum_limits.K,
            sum_limits.Q,
            sum_limits.Jl,
            sum_limits.Jʹl,
            sum_limits.Ju,
            sum_limits.Ku,
            sum_limits.Qu,
            sum_limits.Jʹu,
            sum_limits.Jʹʹu,
            ...,  # Ellipsis means reduce all remaining indexes
        )
        result = h_erg_s * self.nu / 4 / pi * self.N * result
        return result

    @staticmethod
    def compute_epsilon(eta_s: np.ndarray, nu: np.ndarray) -> np.ndarray:
        """
        Compute epsilon given etaS

        Reference: (7.47e)
        """
        return 2 * h_erg_s * nu**3 / c_cm_sm1**2 * np.real(eta_s)

    def create_base_frame(self) -> pd.DataFrame:
        """
        Generate a base frame, listing all transitions. This frame will be used as a starting point to determine
        the ranges for all other summation indexes.
        :return: base frame
        """
        rows = []
        for transition in self.transition_registry.transitions.values():
            term_upper = transition.term_upper
            term_lower = transition.term_lower

            logging.debug(f"Processing {term_upper.term_id} -> {term_lower.term_id}")
            if self.cutoff_condition(term_upper=term_upper, term_lower=term_lower, nu=self.nu):
                logging.info(
                    f"Cutting off the transition {term_upper.term_id} -> {term_lower.term_id} "
                    f"because it does not contribute to the specified frequency range"
                )
                continue

            row_dict = {
                "transition_id": transition.transition_id,
                "einstein_b_lu": transition.einstein_b_lu,
                "einstein_b_ul": transition.einstein_b_ul,
                "einstein_a_ul": transition.einstein_a_ul,
                "term_upper_id": term_upper.term_id,
                "term_lower_id": term_lower.term_id,
                "Ll": term_lower.L,
                "Lu": term_upper.L,
                "S": term_lower.S,
            }
            if self.j_constrained:
                row_dict["lower_J_constraint"] = tuple(transition.lower_J_for_RTE)
                row_dict["upper_J_constraint"] = tuple(transition.upper_J_for_RTE)
            rows.append(row_dict)

        base_frame = pd.DataFrame(rows)
        return base_frame

    @log_method
    def calculate_all_coefficients(
        self, atmosphere_parameters: AtmosphereParameters, angles: Angles, rho: Rho
    ) -> RadiativeTransferCoefficients:
        """
        Compute all radiative transfer coefficients.

        Reference: (7.47)

        :param angles:  Angles instance with LOS and magnetic field angles
        :param rho:  density tensor Rho
        :param atmosphere_parameters:  AtmosphereParameters instance
        :return: RadiativeTransferCoefficients instance
        """
        eta_rho_aI = self.calculate_eta_rho_a(
            stokes_component_index=0,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_aQ = self.calculate_eta_rho_a(
            stokes_component_index=1,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_aU = self.calculate_eta_rho_a(
            stokes_component_index=2,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_aV = self.calculate_eta_rho_a(
            stokes_component_index=3,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )

        eta_rho_sI = self.calculate_eta_rho_s(
            stokes_component_index=0,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_sQ = self.calculate_eta_rho_s(
            stokes_component_index=1,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_sU = self.calculate_eta_rho_s(
            stokes_component_index=2,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_sV = self.calculate_eta_rho_s(
            stokes_component_index=3,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )

        epsilonI = self.compute_epsilon(eta_s=eta_rho_sI, nu=self.nu)
        epsilonQ = self.compute_epsilon(eta_s=eta_rho_sQ, nu=self.nu)
        epsilonU = self.compute_epsilon(eta_s=eta_rho_sU, nu=self.nu)
        epsilonV = self.compute_epsilon(eta_s=eta_rho_sV, nu=self.nu)

        return RadiativeTransferCoefficients(
            eta_rho_aI=eta_rho_aI,
            eta_rho_aQ=eta_rho_aQ,
            eta_rho_aU=eta_rho_aU,
            eta_rho_aV=eta_rho_aV,
            eta_rho_sI=eta_rho_sI,
            eta_rho_sQ=eta_rho_sQ,
            eta_rho_sU=eta_rho_sU,
            eta_rho_sV=eta_rho_sV,
            epsilonI=epsilonI,
            epsilonQ=epsilonQ,
            epsilonU=epsilonU,
            epsilonV=epsilonV,
        )

    @staticmethod
    def phi(
        nui: float, nu: np.ndarray, macroscopic_velocity_cm_sm1: float, delta_v_thermal_cm_sm1: float, voigt_a: float
    ) -> np.ndarray:
        """
        Compute the complex Faraday-Voigt profile.
        delta_v_thermal_cm_sm1 already includes turbulent velocity.

        Reference: (5.43 - 5.45)
        """
        delta_nu_D = nui * delta_v_thermal_cm_sm1 / c_cm_sm1  # Doppler width

        nu_round = (nui - nu) / delta_nu_D  # nui already accounts for magnetic shifts
        nu_round_A = macroscopic_velocity_cm_sm1 / delta_v_thermal_cm_sm1

        complex_voigt = voigt(nu=nu_round - nu_round_A, a=voigt_a) / sqrt_pi / delta_nu_D
        return complex_voigt

    def cutoff_condition(self, term_upper: Term, term_lower: Term, nu: np.ndarray):
        """
        Check the cut-off condition. If a transition is way outside the spectral region of interest,
        it does not contribute to RTE (due to the phi profile).
        """
        nuimax = energy_cmm1_to_frequency_hz(term_upper.get_max_energy_cmm1() - term_lower.get_min_energy_cmm1())
        nuimin = energy_cmm1_to_frequency_hz(term_upper.get_min_energy_cmm1() - term_lower.get_max_energy_cmm1())
        cutoff = self.delta_nu_cutoff
        if min(nu) > nuimax + cutoff or max(nu) < nuimin - cutoff:
            logging.info(f"Cutoff condition: nui=[{nuimin}...{nuimax}], nu=[{min(nu)}...{max(nu)}]")
            return True
        return False

    """
    Summation limits classes:
    These classes control the limits for the summation indexes. We start from the 'base_frame' which has some
    indexes and quantities already pre-merged, like Ll, Lu, S, Einstein coefficients.
    Then we can determine the boundaries of the summation indexes that follow.

    Triangular means from |a-b| to a + b (both ends included)
    FromTo means from a to b (both ends included)
    Intersection means including only shared values of 2 or more sets of values.
    For further information inspect each Looper individually.
    """

    class AFrameSumLimits(SumLimits):
        """
        Summation limits for the eta_A and rho_A calculation.
        """

        term_lower_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        term_upper_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        Ll = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        Lu = DummyOrAlreadyMerged(term_upper_id)  # Pre-merged to base_frame
        S = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        einstein_b_lu = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        jl = Triangular(Ll, S)
        Jl = Triangular(Ll, S)
        Jʹl = Triangular(Ll, S)
        Jʹʹl = Triangular(Ll, S)
        ju = Triangular(Lu, S)
        Ju = Intersection(Triangular(Lu, S), Triangular(Jl, 1))
        Jʹu = Intersection(Triangular(Lu, S), Triangular(Jʹl, 1))
        Ml = Intersection(Projection(Jl), Projection(Jʹʹl), Projection(jl))
        Mʹl = Projection(Jʹl)
        Mu = Intersection(Projection(Ju), Projection(Jʹu), Projection(ju))
        K = FromTo(0, 2)
        Kl = Triangular(Jʹl, Jʹʹl)
        Ql = Intersection(Projection(Kl), Ml - Mʹl)
        q = Ml - Mu
        qʹ = Mʹl - Mu
        Q = Intersection(Projection(K), q - qʹ)

    class AFrameSumLimitsConstrained(SumLimits):
        """
        Summation limits for the eta_A and rho_A calculation, with constraint on J.
        """

        term_lower_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        term_upper_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        Ll = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        Lu = DummyOrAlreadyMerged(term_upper_id)  # Pre-merged to base_frame
        S = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        einstein_b_lu = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        lower_J_constraint = Constraint()  # Artificial constraint for J to model only selected transitions
        upper_J_constraint = Constraint()  # Artificial constraint for J to model only selected transitions
        jl = ApplyConstraint(Triangular(Ll, S), lower_J_constraint)
        Jl = Triangular(Ll, S)
        Jʹl = Triangular(Ll, S)
        Jʹʹl = Triangular(Ll, S)
        ju = ApplyConstraint(Triangular(Lu, S), upper_J_constraint)
        Ju = Intersection(Triangular(Lu, S), Triangular(Jl, 1))
        Jʹu = Intersection(Triangular(Lu, S), Triangular(Jʹl, 1))
        Ml = Intersection(Projection(Jl), Projection(Jʹʹl), Projection(jl))
        Mʹl = Projection(Jʹl)
        Mu = Intersection(Projection(Ju), Projection(Jʹu), Projection(ju))
        K = FromTo(0, 2)
        Kl = Triangular(Jʹl, Jʹʹl)
        Ql = Intersection(Projection(Kl), Ml - Mʹl)
        q = Ml - Mu
        qʹ = Mʹl - Mu
        Q = Intersection(Projection(K), q - qʹ)

    class SFrameSumLimits(SumLimits):
        """
        Summation limits for the eta_S and rho_S calculation.
        """

        term_lower_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        term_upper_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        Ll = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        Lu = DummyOrAlreadyMerged(term_upper_id)  # Pre-merged to base_frame
        S = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        einstein_b_ul = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        ju = Triangular(Lu, S)
        Ju = Triangular(Lu, S)
        Jʹu = Triangular(Lu, S)
        Jʹʹu = Triangular(Lu, S)
        jl = Triangular(Ll, S)
        Jl = Intersection(Triangular(Ll, S), Triangular(Ju, 1))
        Jʹl = Intersection(Triangular(Ll, S), Triangular(Jʹu, 1))
        Mu = Intersection(Projection(Ju), Projection(Jʹʹu), Projection(ju))
        Mʹu = Projection(Jʹu)
        Ml = Intersection(Projection(Jl), Projection(Jʹl), Projection(jl))
        K = FromTo(0, 2)
        Ku = Triangular(Jʹu, Jʹʹu)
        Qu = Intersection(Projection(Ku), Mʹu - Mu)
        q = Ml - Mu
        qʹ = Ml - Mʹu
        Q = Intersection(Projection(K), q - qʹ)

    class SFrameSumLimitsConstrained(SumLimits):
        """
        Summation limits for the eta_S and rho_S calculation, with constraint on J.
        """

        term_lower_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        term_upper_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        Ll = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        Lu = DummyOrAlreadyMerged(term_upper_id)  # Pre-merged to base_frame
        S = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        einstein_b_ul = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        lower_J_constraint = Constraint()  # Artificial constraint for J to model only selected transitions
        upper_J_constraint = Constraint()  # Artificial constraint for J to model only selected transitions
        ju = ApplyConstraint(Triangular(Lu, S), upper_J_constraint)
        Ju = Triangular(Lu, S)
        Jʹu = Triangular(Lu, S)
        Jʹʹu = Triangular(Lu, S)
        jl = ApplyConstraint(Triangular(Ll, S), lower_J_constraint)
        Jl = Intersection(Triangular(Ll, S), Triangular(Ju, 1))
        Jʹl = Intersection(Triangular(Ll, S), Triangular(Jʹu, 1))
        Mu = Intersection(Projection(Ju), Projection(Jʹʹu), Projection(ju))
        Mʹu = Projection(Jʹu)
        Ml = Intersection(Projection(Jl), Projection(Jʹl), Projection(jl))
        K = FromTo(0, 2)
        Ku = Triangular(Jʹu, Jʹʹu)
        Qu = Intersection(Projection(Ku), Mʹu - Mu)
        q = Ml - Mu
        qʹ = Ml - Mʹu
        Q = Intersection(Projection(K), q - qʹ)
