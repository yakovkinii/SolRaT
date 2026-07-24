try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from numpy import pi, sqrt

from solrat.atom_model.base_atom_model.radiative_transfer_equations import BaseRTE
from solrat.atom_model.multi_level_atom_model.object.atmosphere_parameters import AtmosphereParameters
from solrat.atom_model.multi_level_atom_model.object.level_registry import Level, LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.rho_matrix_builder import Rho
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.radiative_transfer_coefficients import RadiativeTransferCoefficients
from solrat.atom_model.shared.object.rotations import T_K_Q_double_rotation_all_stokes, WignerD
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, sqrt_pi
from solrat.atom_model.shared.utility.functions import energy_cmm1_to_frequency_sm1
from solrat.atom_model.shared.utility.voigt_profile import voigt
from solrat.atom_model.shared.utility.wigner_3j_6j_9j import wigner_3j
from solrat.engine.functions.decorators import VERBOSE, log_method
from solrat.engine.functions.general import m1p, n_proj
from solrat.engine.generators.merge_frame import Frame, SumLimits
from solrat.engine.generators.merge_loopers import DummyOrAlreadyMerged, FromTo, Intersection, Projection, Triangular


class MultiLevelAtomRTE(BaseRTE):
    r"""
    Radiative Transfer Coefficients within the Multi-Level atom model.

    :param level_registry:  :class:`LevelRegistry` instance.
    :param transition_registry:  :class:`TransitionRegistry` instance.
    :param nu:  frequencies [Hz].
    :param custom_delta_nu_cutoff:  distance in frequency for cutting off irrelevant transitions.
        Leave None for a conservative default value.
    :param N:  atom numeric concentration for d/dz transfer modeling. Can be left equal to 1 for d/dtau modeling.

    Reference: (LL04 7.15)
    """

    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        nu: np.ndarray,
        custom_delta_nu_cutoff=None,
        N: float = 1.0,
    ):
        self.level_registry: LevelRegistry = level_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.nu = nu
        self.delta_nu_cutoff = (
            custom_delta_nu_cutoff
            if custom_delta_nu_cutoff is not None
            else max(10 * (np.max(nu) - np.min(nu)), np.mean(nu) * 1e-3)
        )
        self.N = N

        # Shorter getters for Einstein coefficients
        self.einstein_b_lu = np.vectorize(self.transition_registry.einstein_b_lu)
        self.einstein_b_ul = np.vectorize(self.transition_registry.einstein_b_ul)

        # Precomputed frames: atom-specific angular algebra (atmosphere-independent).
        self.eta_rho_a_frame: Union[Frame, None] = None
        self.eta_rho_s_frame: Union[Frame, None] = None
        # Per-(angles, atmosphere) cache of the rho-index-reduced operator frames. Opt-in
        # (off by default): worthwhile when calculate_eta_rho_* is called repeatedly with the same
        # geometry/atmosphere and only rho varying (e.g. the NLTE iteration). Enable deliberately
        # and clear_operator_cache() when done; the atom-level frame caches above are always on.
        self.use_operator_cache: bool = False
        self.eta_rho_a_operator_cache: Dict = {}
        self.eta_rho_s_operator_cache: Dict = {}

    @classmethod
    def from_model_config(
        cls,
        config: MultiLevelAtomConfig,
        nu: np.ndarray,
    ) -> Self:
        logging.info("Constructing MultiLevelAtomRTE instance")
        return cls(
            level_registry=config.level_registry,
            transition_registry=config.transition_registry,
            nu=nu,
            custom_delta_nu_cutoff=config.custom_delta_nu_cutoff,
            N=config.N,
        )

    def clear_operator_cache(self) -> None:
        r"""
        Empty the opt-in per-(angles, atmosphere) operator caches to free memory.
        The atom-level frame caches (eta_rho_a_frame / eta_rho_s_frame) are kept.
        """
        self.eta_rho_a_operator_cache.clear()
        self.eta_rho_s_operator_cache.clear()

    @log_method
    def calculate_eta_rho_a(self, angles: Angles, rho: Rho, atmosphere_parameters: AtmosphereParameters) -> np.ndarray:
        r"""
        Calculate :math:`\eta_A + i \rho_A` for all Stokes components simultaneously.

        :return: complex array of shape ``[4, len(nu)]`` for I, Q, U, V.

        Reference: (LL04 7.15 ac)
        """
        rho_index_cols = ["level_lower_id", "Kl", "Ql"]
        cache_key = (
            angles.chi, angles.theta, angles.gamma, angles.chi_B, angles.theta_B,
            atmosphere_parameters.magnetic_field_gauss,
            atmosphere_parameters.macroscopic_velocity_cm_sm1,
            atmosphere_parameters.delta_v_thermal_cm_sm1,
            atmosphere_parameters.voigt_a,
        )  # fmt: skip

        if self.use_operator_cache and cache_key in self.eta_rho_a_operator_cache:
            frame = self.eta_rho_a_operator_cache[cache_key].copy()
        else:
            if self.eta_rho_a_frame is None:
                frame = Frame.from_sum_limits(
                    base_frame=self.create_base_frame(),
                    sum_limits=self.AFrameSumLimits(),
                )
                frame.register_multiplication(
                    a001=lambda Jl:                          (2 * Jl + 1),
                    a002=lambda transition_id, K, Kl:        self.einstein_b_lu(transition_id) * sqrt(n_proj(1, K, Kl)),
                    a003=lambda Jl, Ml, qʹ:                  m1p(1 + Jl - Ml + qʹ),
                    w3j1=lambda Ju, Jl, Mu, Ml, q:           wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    w3j2=lambda Ju, Jl, Mu, Mʹl, qʹ:         wigner_3j(Ju, Jl, 1, -Mu, Mʹl, -qʹ),
                    w3j3=lambda K, q, qʹ, Q:                 wigner_3j(1, 1, K, q, -qʹ, -Q),
                    w3j4=lambda Jl, Kl, Ml, Mʹl, Ql:         wigner_3j(Jl, Jl, Kl, Ml, -Mʹl, -Ql),
                )  # fmt: skip
                self.eta_rho_a_frame = frame.copy()
            else:
                frame = self.eta_rho_a_frame.copy()

            # Angle / field / profile factors (everything except rho), then reduce to the rho indexes.
            D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
            D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)
            frame.register_multiplication(
                tkq=lambda K, Q: T_K_Q_double_rotation_all_stokes(
                    K=K, Q=Q, D_inverse_omega=D_inverse_omega, D_magnetic=D_magnetic
                ),
                phi=lambda level_upper_id, Mu, level_lower_id, Ml, gu, gl: self.phi(
                    level_upper_id=level_upper_id,
                    Mu=Mu,
                    level_lower_id=level_lower_id,
                    Ml=Ml,
                    gu=gu,
                    gl=gl,
                    atmosphere_parameters=atmosphere_parameters,
                ),
                elementwise=True,
            )
            # Reduce every column except the rho-index columns (level_lower_id, Kl, Ql), heavy K, Q first.
            reduce_cols = [
                "K", "Q", "Ju", "Jl",
                "transition_id", "level_upper_id", "gl", "gu", "Ml", "Mʹl", "Mu", "q", "qʹ",
            ]  # fmt: skip
            frame.reduce_partially(*reduce_cols)
            if self.use_operator_cache:
                self.eta_rho_a_operator_cache[cache_key] = frame.copy()

        # Per-call: multiply by rho and reduce over the rho indexes only.
        frame.register_multiplication(
            rho=lambda level_lower_id, Kl, Ql: rho(level_id=level_lower_id, K=Kl, Q=Ql),
            elementwise=True,
        )
        result = frame.reduce(*rho_index_cols)
        result = h_erg_s * self.nu / 4 / pi * self.N * result
        return result

    @log_method
    def calculate_eta_rho_s(self, angles: Angles, rho: Rho, atmosphere_parameters: AtmosphereParameters) -> np.ndarray:
        r"""
        Calculate :math:`\eta_S + i \rho_S` for all Stokes components simultaneously.

        :return: complex array of shape ``[4, len(nu)]`` for I, Q, U, V.

        Reference: (LL04 7.15 bd)
        """
        rho_index_cols = ["level_upper_id", "Ku", "Qu"]
        cache_key = (
            angles.chi, angles.theta, angles.gamma, angles.chi_B, angles.theta_B,
            atmosphere_parameters.magnetic_field_gauss,
            atmosphere_parameters.macroscopic_velocity_cm_sm1,
            atmosphere_parameters.delta_v_thermal_cm_sm1,
            atmosphere_parameters.voigt_a,
        )  # fmt: skip

        if self.use_operator_cache and cache_key in self.eta_rho_s_operator_cache:
            frame = self.eta_rho_s_operator_cache[cache_key].copy()
        else:
            if self.eta_rho_s_frame is None:
                frame = Frame.from_sum_limits(
                    base_frame=self.create_base_frame(),
                    sum_limits=self.SFrameSumLimits(),
                )
                frame.register_multiplication(
                    a001=lambda Ju:                          (2 * Ju + 1),
                    a002=lambda transition_id, K, Ku:        self.einstein_b_ul(transition_id) * sqrt(n_proj(1, K, Ku)),
                    a003=lambda Ju, Mu, qʹ:                  m1p(1 + Ju - Mu + qʹ),
                    w3j1=lambda Ju, Jl, Mu, Ml, q:           wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    w3j2=lambda Ju, Jl, Mʹu, Ml, qʹ:         wigner_3j(Ju, Jl, 1, -Mʹu, Ml, -qʹ),
                    w3j3=lambda K, q, qʹ, Q:                 wigner_3j(1, 1, K, q, -qʹ, -Q),
                    w3j4=lambda Ju, Ku, Mu, Mʹu, Qu:         wigner_3j(Ju, Ju, Ku, Mʹu, -Mu, -Qu),
                )  # fmt: skip
                self.eta_rho_s_frame = frame.copy()
            else:
                frame = self.eta_rho_s_frame.copy()

            # Angle / field / profile factors (everything except rho), then reduce to the rho indexes.
            D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
            D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)
            frame.register_multiplication(
                tkq=lambda K, Q: T_K_Q_double_rotation_all_stokes(
                    K=K, Q=Q, D_inverse_omega=D_inverse_omega, D_magnetic=D_magnetic
                ),
                phi=lambda level_upper_id, Mu, level_lower_id, Ml, gu, gl: self.phi(
                    level_upper_id=level_upper_id,
                    Mu=Mu,
                    level_lower_id=level_lower_id,
                    Ml=Ml,
                    gu=gu,
                    gl=gl,
                    atmosphere_parameters=atmosphere_parameters,
                ),
                elementwise=True,
            )
            # Reduce every column except the rho-index columns (level_upper_id, Ku, Qu), heavy K, Q first.
            reduce_cols = [
                "K", "Q", "Jl", "Ju",
                "transition_id", "level_lower_id", "gl", "gu", "Mu", "Mʹu", "Ml", "q", "qʹ",
            ]  # fmt: skip
            frame.reduce_partially(*reduce_cols)
            if self.use_operator_cache:
                self.eta_rho_s_operator_cache[cache_key] = frame.copy()

        # Per-call: multiply by rho and reduce over the rho indexes only.
        frame.register_multiplication(
            rho=lambda level_upper_id, Ku, Qu: rho(level_id=level_upper_id, K=Ku, Q=Qu),
            elementwise=True,
        )
        result = frame.reduce(*rho_index_cols)
        result = h_erg_s * self.nu / 4 / pi * self.N * result
        return result

    @staticmethod
    def calculate_epsilon(eta_s: np.ndarray, nu: np.ndarray) -> np.ndarray:
        r"""
        Compute :math:`\epsilon` given :math:`\eta_S`.

        Reference: (LL04 7.15e)
        """
        return 2 * h_erg_s * nu**3 / c_cm_sm1**2 * np.real(eta_s)

    @log_method
    def create_base_frame(self) -> pd.DataFrame:
        r"""
        Generate a base frame, listing all transitions.
        """
        rows = []
        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower

            logging.debug(f"Processing {level_upper.level_id} -> {level_lower.level_id}")
            if self.cutoff_condition(level_upper=level_upper, level_lower=level_lower, nu=self.nu):
                logging.log(
                    VERBOSE,
                    f"Cutting off the transition {level_upper.level_id} -> {level_lower.level_id} "
                    f"because it does not contribute to the specified frequency range",
                )
                continue

            rows.append(
                {
                    "transition_id": transition.transition_id,
                    "level_upper_id": level_upper.level_id,
                    "level_lower_id": level_lower.level_id,
                    "Jl": level_lower.J,
                    "Ju": level_upper.J,
                    "gl": level_lower.g,
                    "gu": level_upper.g,
                }
            )
        if not rows:
            raise ValueError(
                "No registered transitions contribute to the supplied frequency grid. "
                "Check that the frequency grid contains an actual transition frequency."
            )
        return pd.DataFrame(rows)

    @log_method
    def calculate_all_coefficients(
        self, atmosphere_parameters: AtmosphereParameters, angles: Angles, rho: Rho
    ) -> RadiativeTransferCoefficients:
        r"""
        Compute all radiative transfer coefficients.

        Reference: (LL04 7.15)
        """
        # logging.info("Calculating Radiative Transfer Coefficients")

        eta_rho_a = self.calculate_eta_rho_a(
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )

        eta_rho_s = self.calculate_eta_rho_s(
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )

        return RadiativeTransferCoefficients(
            eta_rho_aI=eta_rho_a[0],
            eta_rho_aQ=eta_rho_a[1],
            eta_rho_aU=eta_rho_a[2],
            eta_rho_aV=eta_rho_a[3],
            eta_rho_sI=eta_rho_s[0],
            eta_rho_sQ=eta_rho_s[1],
            eta_rho_sU=eta_rho_s[2],
            eta_rho_sV=eta_rho_s[3],
            epsilonI=self.calculate_epsilon(eta_s=eta_rho_s[0], nu=self.nu),
            epsilonQ=self.calculate_epsilon(eta_s=eta_rho_s[1], nu=self.nu),
            epsilonU=self.calculate_epsilon(eta_s=eta_rho_s[2], nu=self.nu),
            epsilonV=self.calculate_epsilon(eta_s=eta_rho_s[3], nu=self.nu),
        )

    @staticmethod
    def _phi(
        nui: float, nu: np.ndarray, macroscopic_velocity_cm_sm1: float, delta_v_thermal_cm_sm1: float, voigt_a: float
    ) -> np.ndarray:
        """
        Complex Faraday-Voigt profile.

        Reference: (LL04 5.43 - 5.45)
        """
        delta_nu_D = nui * delta_v_thermal_cm_sm1 / c_cm_sm1
        nu_round = (nui - nu) / delta_nu_D
        nu_round_A = macroscopic_velocity_cm_sm1 / delta_v_thermal_cm_sm1
        complex_voigt = voigt(nu=nu_round - nu_round_A, a=voigt_a) / sqrt_pi / delta_nu_D
        return complex_voigt

    def phi(self, level_upper_id, Mu, level_lower_id, Ml, gu, gl, atmosphere_parameters):
        r"""
        Doppler/Voigt profile evaluated at the linear-Zeeman-shifted frequency
        :math:`\nu_{\alpha_u J_u M_u, \alpha_l J_l M_l}`.
        """
        nu_0 = energy_cmm1_to_frequency_sm1(
            self.level_registry.levels[level_upper_id].energy_cmm1
            - self.level_registry.levels[level_lower_id].energy_cmm1
        )
        nui = nu_0 + (gu * Mu - gl * Ml) * atmosphere_parameters.nu_larmor
        return self._phi(
            nui=nui,
            nu=self.nu,
            macroscopic_velocity_cm_sm1=atmosphere_parameters.macroscopic_velocity_cm_sm1,
            delta_v_thermal_cm_sm1=atmosphere_parameters.delta_v_thermal_cm_sm1,
            voigt_a=atmosphere_parameters.voigt_a,
        )

    def cutoff_condition(self, level_upper: Level, level_lower: Level, nu: np.ndarray):
        r"""
        Cut-off condition for transitions far from the spectral region of interest.
        """
        nui = energy_cmm1_to_frequency_sm1(level_upper.energy_cmm1 - level_lower.energy_cmm1)
        cutoff = self.delta_nu_cutoff
        if min(nu) > nui + cutoff or max(nu) < nui - cutoff:
            logging.log(VERBOSE, f"Cutoff condition: nui={nui}, nu=[{min(nu)}...{max(nu)}]")
            return True
        return False

    class AFrameSumLimits(SumLimits):
        r"""
        Summation limits for the :math:`\eta_A` and :math:`\rho_A` calculation (eq. 7.15a).
        """

        transition_id = DummyOrAlreadyMerged()
        level_lower_id = DummyOrAlreadyMerged()
        level_upper_id = DummyOrAlreadyMerged()
        Jl = DummyOrAlreadyMerged(level_lower_id)
        Ju = DummyOrAlreadyMerged(level_upper_id)
        gl = DummyOrAlreadyMerged(level_lower_id)
        gu = DummyOrAlreadyMerged(level_upper_id)
        Ml = Projection(Jl)
        Mʹl = Projection(Jl)
        Mu = Projection(Ju)
        K = FromTo(0, 2)
        Kl = Triangular(Jl, Jl)
        Ql = Intersection(Projection(Kl), Ml - Mʹl)
        q = Ml - Mu
        qʹ = Mʹl - Mu
        Q = Intersection(Projection(K), q - qʹ)

    class SFrameSumLimits(SumLimits):
        r"""
        Summation limits for the :math:`\eta_S` and :math:`\rho_S` calculation (eq. 7.15b).
        """

        transition_id = DummyOrAlreadyMerged()
        level_lower_id = DummyOrAlreadyMerged()
        level_upper_id = DummyOrAlreadyMerged()
        Jl = DummyOrAlreadyMerged(level_lower_id)
        Ju = DummyOrAlreadyMerged(level_upper_id)
        gl = DummyOrAlreadyMerged(level_lower_id)
        gu = DummyOrAlreadyMerged(level_upper_id)
        Mu = Projection(Ju)
        Mʹu = Projection(Ju)
        Ml = Projection(Jl)
        K = FromTo(0, 2)
        Ku = Triangular(Ju, Ju)
        Qu = Intersection(Projection(Ku), Mʹu - Mu)
        q = Ml - Mu
        qʹ = Ml - Mʹu
        Q = Intersection(Projection(K), q - qʹ)
