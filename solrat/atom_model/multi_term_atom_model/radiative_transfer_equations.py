try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import logging
from typing import Union

import numpy as np
import pandas as pd
from numpy import pi, sqrt

from solrat.atom_model.base_atom_model.radiative_transfer_equations import BaseRTE
from solrat.atom_model.multi_term_atom_model.object.atmosphere_parameters import AtmosphereParameters
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry, Term
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.rho_matrix_builder import Rho
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.multi_term_atom_model.utility.paschen_back import PaschenBack
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.radiative_transfer_coefficients import RadiativeTransferCoefficients
from solrat.atom_model.shared.object.rotations import T_K_Q_double_rotation_all_stokes, WignerD
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, sqrt_pi
from solrat.atom_model.shared.utility.functions import energy_cmm1_to_frequency_sm1
from solrat.atom_model.shared.utility.voigt_profile import voigt
from solrat.atom_model.shared.utility.wigner_3j_6j_9j import wigner_3j, wigner_6j
from solrat.engine.functions.decorators import VERBOSE, log_method
from solrat.engine.functions.general import m1p, n_proj
from solrat.engine.generators.merge_frame import Frame, SumLimits
from solrat.engine.generators.merge_loopers import (
    ApplyConstraint,
    Constraint,
    DummyOrAlreadyMerged,
    FromTo,
    Intersection,
    Projection,
    Triangular,
)


class MultiTermAtomRTE(BaseRTE):
    r"""
    Radiative Transfer Coefficients within Multi-Term atom model.

    :param level_registry:  LevelRegistry instance for the multi-term atom under study.
    :param transition_registry:  TransitionRegistry instance for the multi-term atom under study.
    :param nu:  frequencies [Hz]
    :param custom_delta_nu_cutoff:  distance in frequency for cutting off irrelevant transitions.
        Leave None for a conservative default value.
    :param N:  atom numeric concentration for d/dz transfer modeling. Can be left equal to 1 for d/dtau modeling.
    :param j_constrained:  constrain J values to the ones specified in transition_registry.
        This parameter is useful for modeling lines like Fe5434 where fine structure components are scattered
        over a very broad spectral interval, while the user is interested only in a specific transition.

    Reference: (LL04 7.47)
    """

    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        nu: np.ndarray,
        custom_delta_nu_cutoff=None,
        N: float = 1.0,
        j_constrained=False,
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
        self.j_constrained = j_constrained

        # Shorter getters for Einstein coefficients
        self.paschen_back = PaschenBack(level_registry=level_registry)
        self.einstein_b_lu = np.vectorize(self.transition_registry.einstein_b_lu)
        self.einstein_b_ul = np.vectorize(self.transition_registry.einstein_b_ul)

        # Precomputed frames
        self.eta_rho_a_frame: Union[Frame, None] = None
        self.eta_rho_s_frame: Union[Frame, None] = None

    @classmethod
    def from_model_config(
        cls,
        config: MultiTermAtomConfig,
        nu: np.ndarray,
    ) -> Self:
        r"""
        Constructor from the model config.
        """
        logging.info("Constructing MultiTermAtomRTE instance")

        return cls(
            level_registry=config.level_registry,
            transition_registry=config.transition_registry,
            nu=nu,
            custom_delta_nu_cutoff=config.custom_delta_nu_cutoff,
            N=config.N,
            j_constrained=config.j_constrained,
        )

    @log_method
    def calculate_eta_rho_a(self, angles: Angles, rho: Rho, atmosphere_parameters: AtmosphereParameters) -> np.ndarray:
        r"""
        Calculate etaA and rhoA for all Stokes components simultaneously.

        :param angles:  Angles instance with LOS and magnetic field angles
        :param rho:  density tensor Rho
        :param atmosphere_parameters:  AtmosphereParameters instance
        :return: complex array of :math:`\eta_A + i \rho_A` vs frequency, shape [4, len(nu)] for I, Q, U, V

        Reference: (LL04 7.47 ac)
        """
        magnetic_field_gauss = atmosphere_parameters.magnetic_field_gauss

        if self.eta_rho_a_frame is None:
            # If no cached frame available, calculate atmosphere-independent part and save
            frame = Frame.from_sum_limits(
                base_frame=self.create_base_frame(),
                sum_limits=self.AFrameSumLimitsConstrained() if self.j_constrained else self.AFrameSumLimits(),
            )

            frame.register_multiplication(
                a001=lambda Ll:                          n_proj(Ll),
                a002=lambda transition_id, K, Kl:        self.einstein_b_lu(transition_id) * sqrt(n_proj(1, K, Kl)),
                a003=lambda Jʹʹl, Ml, qʹ:                m1p(1 + Jʹʹl - Ml + qʹ),
                a004=lambda Jl, Jʹl, Ju, Jʹu:            sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                w3j1=lambda Ju, Jl, Mu, Ml, q:           wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                w3j2=lambda Jʹu, Jʹl, Mu, Mʹl, qʹ:       wigner_3j(Jʹu, Jʹl, 1, -Mu, Mʹl, -qʹ),
                w3j3=lambda K, q, qʹ, Q:                 wigner_3j(1, 1, K, q, -qʹ, -Q),
                w3j4=lambda Jʹʹl, Jʹl, Kl, Ml, Mʹl, Ql:  wigner_3j(Jʹʹl, Jʹl, Kl, Ml, -Mʹl, -Ql),
                w6j1=lambda Lu, Ll, Jl, Ju, S:           wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                w6j2=lambda Lu, Ll, Jʹl, Jʹu, S:         wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
            )  # fmt: skip

            self.eta_rho_a_frame = frame.copy()
        else:
            frame = self.eta_rho_a_frame.copy()

        D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
        D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)

        frame.register_multiplication(
            tkq=lambda K, Q: T_K_Q_double_rotation_all_stokes(
                K=K, Q=Q, D_inverse_omega=D_inverse_omega, D_magnetic=D_magnetic
            ),
            pb1=lambda term_lower_id, jl, Jl, Ml: self.paschen_back.eigenvector(
                term_id=term_lower_id, j=jl, J=Jl, M=Ml, magnetic_field_gauss=magnetic_field_gauss
            ),
            pb2=lambda term_lower_id, jl, Jʹʹl, Ml: self.paschen_back.eigenvector(
                term_id=term_lower_id, j=jl, J=Jʹʹl, M=Ml, magnetic_field_gauss=magnetic_field_gauss
            ),
            pb3=lambda term_upper_id, ju, Ju, Mu: self.paschen_back.eigenvector(
                term_id=term_upper_id, j=ju, J=Ju, M=Mu, magnetic_field_gauss=magnetic_field_gauss
            ),
            pb4=lambda term_upper_id, ju, Jʹu, Mu: self.paschen_back.eigenvector(
                term_id=term_upper_id, j=ju, J=Jʹu, M=Mu, magnetic_field_gauss=magnetic_field_gauss
            ),
            rho=lambda term_lower_id, Kl, Ql, Jʹʹl, Jʹl: rho(term_id=term_lower_id, K=Kl, Q=Ql, J=Jʹʹl, Jʹ=Jʹl),
            phi=lambda ju, Mu, term_upper_id, jl, Ml, term_lower_id: self.phi(
                term_upper_id=term_upper_id,
                ju=ju,
                Mu=Mu,
                term_lower_id=term_lower_id,
                jl=jl,
                Ml=Ml,
                atmosphere_parameters=atmosphere_parameters,
            ),
            elementwise=True,
        )

        result = frame.reduce("K", "Q", "Ju", "Jʹu", "Jl", "Kl", "Ql", "Jʹʹl", "Jʹl", ...)
        result = h_erg_s * self.nu / 4 / pi * self.N * result
        return result

    @log_method
    def calculate_eta_rho_s(self, angles: Angles, rho: Rho, atmosphere_parameters: AtmosphereParameters) -> np.ndarray:
        r"""
        Calculate etaS and rhoS for all Stokes components simultaneously.

        :param angles:  Angles instance with LOS and magnetic field angles
        :param rho:  density tensor Rho
        :param atmosphere_parameters:  AtmosphereParameters instance
        :return: complex array of :math:`\eta_S + i \rho_S` vs frequency, shape [4, len(nu)] for I, Q, U, V

        Reference: (LL04 7.47 bd)
        """
        magnetic_field_gauss = atmosphere_parameters.magnetic_field_gauss

        if self.eta_rho_s_frame is None:
            # If no cached frame available, calculate atmosphere-independent part and save

            frame = Frame.from_sum_limits(
                base_frame=self.create_base_frame(),
                sum_limits=self.SFrameSumLimitsConstrained() if self.j_constrained else self.SFrameSumLimits(),
            )

            frame.register_multiplication(
                a001=lambda Lu: n_proj(Lu),
                a002=lambda transition_id, K, Ku: self.einstein_b_ul(transition_id) * sqrt(n_proj(1, K, Ku)),
                a003=lambda Jʹu, Mu, qʹ: m1p(1 + Jʹu - Mu + qʹ),
                a004=lambda Jl, Jʹl, Ju, Jʹu: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                w3j1=lambda Ju, Jl, Mu, Ml, q: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                w3j2=lambda Jʹu, Jʹl, Mʹu, Ml, qʹ: wigner_3j(Jʹu, Jʹl, 1, -Mʹu, Ml, -qʹ),
                w3j3=lambda K, q, qʹ, Q: wigner_3j(1, 1, K, q, -qʹ, -Q),
                w3j4=lambda Jʹʹu, Jʹu, Ku, Mu, Mʹu, Qu: wigner_3j(Jʹu, Jʹʹu, Ku, Mʹu, -Mu, -Qu),
                w6j1=lambda Lu, Ll, Jl, Ju, S: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                w6j2=lambda Lu, Ll, Jʹl, Jʹu, S: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
            )  # fmt: skip
            self.eta_rho_s_frame = frame.copy()
        else:
            frame = self.eta_rho_s_frame.copy()

        D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
        D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)

        frame.register_multiplication(
            tkq=lambda K, Q: T_K_Q_double_rotation_all_stokes(
                K=K, Q=Q, D_inverse_omega=D_inverse_omega, D_magnetic=D_magnetic
            ),
            pb1=lambda term_lower_id, jl, Jl, Ml: self.paschen_back.eigenvector(
                term_id=term_lower_id, j=jl, J=Jl, M=Ml, magnetic_field_gauss=magnetic_field_gauss
            ),
            pb2=lambda term_lower_id, jl, Jʹl, Ml: self.paschen_back.eigenvector(
                term_id=term_lower_id, j=jl, J=Jʹl, M=Ml, magnetic_field_gauss=magnetic_field_gauss
            ),
            pb3=lambda term_upper_id, ju, Ju, Mu: self.paschen_back.eigenvector(
                term_id=term_upper_id, j=ju, J=Ju, M=Mu, magnetic_field_gauss=magnetic_field_gauss
            ),
            pb4=lambda term_upper_id, ju, Jʹʹu, Mu: self.paschen_back.eigenvector(
                term_id=term_upper_id, j=ju, J=Jʹʹu, M=Mu, magnetic_field_gauss=magnetic_field_gauss
            ),
            rho=lambda term_upper_id, Ku, Qu, Jʹʹu, Jʹu: rho(term_id=term_upper_id, K=Ku, Q=Qu, J=Jʹu, Jʹ=Jʹʹu),
            phi=lambda ju, Mu, term_upper_id, jl, Ml, term_lower_id: self.phi(
                term_upper_id=term_upper_id,
                ju=ju,
                Mu=Mu,
                term_lower_id=term_lower_id,
                jl=jl,
                Ml=Ml,
                atmosphere_parameters=atmosphere_parameters,
            ),
            elementwise=True,
        )

        result = frame.reduce("K", "Q", "Jl", "Jʹl", "Ju", "Ku", "Qu", "Jʹu", "Jʹʹu", ...)
        result = h_erg_s * self.nu / 4 / pi * self.N * result
        return result

    @staticmethod
    def calculate_epsilon(eta_s: np.ndarray, nu: np.ndarray) -> np.ndarray:
        r"""
        Compute :math:`\epsilon` given :math:`\eta_S`

        Reference: (LL04 7.47e)
        """
        return 2 * h_erg_s * nu**3 / c_cm_sm1**2 * np.real(eta_s)

    @log_method
    def create_base_frame(self) -> pd.DataFrame:
        r"""
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
                logging.log(
                    VERBOSE,
                    f"Cutting off the transition {term_upper.term_id} -> {term_lower.term_id} "
                    f"because it does not contribute to the specified frequency range",
                )
                continue

            row_dict = {
                "transition_id": transition.transition_id,
                "term_upper_id": term_upper.term_id,
                "term_lower_id": term_lower.term_id,
                "Ll": term_lower.L,
                "Lu": term_upper.L,
                "S": term_lower.S,
            }
            if self.j_constrained:
                assert transition.lower_J_for_RTE is not None
                assert transition.upper_J_for_RTE is not None
                row_dict["lower_J_constraint"] = tuple(transition.lower_J_for_RTE)
                row_dict["upper_J_constraint"] = tuple(transition.upper_J_for_RTE)
            rows.append(row_dict)

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

        :param angles:  Angles instance with LOS and magnetic field angles
        :param rho:  density tensor Rho
        :param atmosphere_parameters:  AtmosphereParameters instance
        :return: RadiativeTransferCoefficients instance

        Reference: (LL04 7.47)
        """
        logging.info("Calculating Radiative Transfer Coefficients")

        eta_rho_a = self.calculate_eta_rho_a(
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )  # shape [4, len(nu)]: I, Q, U, V

        eta_rho_s = self.calculate_eta_rho_s(
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )  # shape [4, len(nu)]: I, Q, U, V

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
        Compute the complex Faraday-Voigt profile.

        delta_v_thermal_cm_sm1 already includes turbulent velocity.

        Reference: (LL04 5.43 - 5.45)
        """
        delta_nu_D = nui * delta_v_thermal_cm_sm1 / c_cm_sm1  # Doppler width

        nu_round = (nui - nu) / delta_nu_D  # nui already accounts for magnetic shifts
        nu_round_A = macroscopic_velocity_cm_sm1 / delta_v_thermal_cm_sm1

        complex_voigt = voigt(nu=nu_round - nu_round_A, a=voigt_a) / sqrt_pi / delta_nu_D
        return complex_voigt

    def phi(self, term_upper_id, ju, Mu, term_lower_id, jl, Ml, atmosphere_parameters):
        return self._phi(
            nui=energy_cmm1_to_frequency_sm1(
                self.paschen_back.eigenvalue(
                    term_id=term_upper_id,
                    j=ju,
                    M=Mu,
                    magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
                )
                - self.paschen_back.eigenvalue(
                    term_id=term_lower_id,
                    j=jl,
                    M=Ml,
                    magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
                )
            ),
            nu=self.nu,
            macroscopic_velocity_cm_sm1=atmosphere_parameters.macroscopic_velocity_cm_sm1,
            delta_v_thermal_cm_sm1=atmosphere_parameters.delta_v_thermal_cm_sm1,
            voigt_a=atmosphere_parameters.voigt_a,
        )

    def cutoff_condition(self, term_upper: Term, term_lower: Term, nu: np.ndarray):
        r"""
        Check the cut-off condition. If a transition is way outside the spectral region of interest,
        it does not contribute to RTE (due to the phi profile).
        """
        nuimax = energy_cmm1_to_frequency_sm1(term_upper.get_max_energy_cmm1() - term_lower.get_min_energy_cmm1())
        nuimin = energy_cmm1_to_frequency_sm1(term_upper.get_min_energy_cmm1() - term_lower.get_max_energy_cmm1())
        cutoff = self.delta_nu_cutoff
        if min(nu) > nuimax + cutoff or max(nu) < nuimin - cutoff:
            logging.log(VERBOSE, f"Cutoff condition: nui=[{nuimin}...{nuimax}], nu=[{min(nu)}...{max(nu)}]")
            return True
        return False

    class AFrameSumLimits(SumLimits):
        r"""
        Summation limits for the :math:`\eta_A` and :math:`\rho_A` calculation.
        See SumLimits for reference.
        """

        term_lower_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        term_upper_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        Ll = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        Lu = DummyOrAlreadyMerged(term_upper_id)  # Pre-merged to base_frame
        S = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
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
        r"""
        Summation limits for the :math:`\eta_A` and :math:`\rho_A` calculation, with constraint on :math:`J`.
        See SumLimits for reference.
        """

        term_lower_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        term_upper_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        Ll = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        Lu = DummyOrAlreadyMerged(term_upper_id)  # Pre-merged to base_frame
        S = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
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
        r"""
        Summation limits for the :math:`\eta_S` and :math:`\rho_S` calculation.
        See SumLimits for reference.
        """

        term_lower_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        term_upper_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        Ll = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        Lu = DummyOrAlreadyMerged(term_upper_id)  # Pre-merged to base_frame
        S = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
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
        r"""
        Summation limits for the :math:`\eta_S` and :math:`\rho_S` calculation, with constraint on :math:`J`.
        See SumLimits for reference.
        """

        term_lower_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        term_upper_id = DummyOrAlreadyMerged()  # Pre-merged to base_frame
        Ll = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
        Lu = DummyOrAlreadyMerged(term_upper_id)  # Pre-merged to base_frame
        S = DummyOrAlreadyMerged(term_lower_id)  # Pre-merged to base_frame
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
