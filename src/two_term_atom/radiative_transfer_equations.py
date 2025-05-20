import logging

import numpy as np
import pandas as pd
from numpy import pi, sqrt

from src.core.engine.functions.decorators import log_method
from src.core.engine.functions.general import m1p, n_proj
from src.core.engine.functions.looping import FROMTO, INTERSECTION, PROJECTION, TRIANGULAR, VALUE
from src.core.engine.generators.multiply import multiply
from src.core.engine.generators.nested_loops import nested_loops
from src.core.physics.constants import c_cm_sm1, h_erg_s, sqrt_pi
from src.core.physics.functions import energy_cmm1_to_frequency_hz
from src.core.physics.rotations import T_K_Q_double_rotation, WignerD
from src.core.physics.voigt_profile import voigt
from src.core.physics.wigner_3j_6j_9j import wigner_3j, wigner_6j
from src.two_term_atom.object.angles import Angles
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.rho_matrix_builder import Rho
from src.two_term_atom.physics.paschen_back import calculate_paschen_back
from src.two_term_atom.terms_levels_transitions.term_registry import Level, TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class TwoTermAtomRTE:
    """
    term_registry: defines all terms within the atom model.
    transition_registry: defines all transitions within the atom model.
    nu: defines the spectral interval of interest for cutting off non-contributing transitions.
    maximum_delta_v_thermal_units_cutoff: defines the cutoff condition for the transition.
    angles: optional, if provided outright, allows to reduce the frame by precomputing T_K_Q tensor
    rho: optional, if provided outright, allows to reduce the frame by precomputing rho matrix
    N: atom concentration
    """

    def __init__(
        self,
        term_registry: TermRegistry,
        transition_registry: TransitionRegistry,
        nu: np.ndarray,
        delta_nu_cutoff=None,
        angles: Angles = None,
        magnetic_field_gauss=None,
        rho: Rho = None,
        N=1,
    ):
        self.term_registry: TermRegistry = term_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.nu = nu
        self.delta_nu_cutoff = delta_nu_cutoff
        if self.delta_nu_cutoff is None:
            self.delta_nu_cutoff = max(10 * (np.max(nu) - np.min(nu)), np.mean(nu) * 1e-3)
        self.N = N  # Atom concentration
        self.frame_s_id_columns = []
        self.frame_s = self.precompute_eta_s_frame()
        self.magnetic_field_gauss = magnetic_field_gauss
        self.t_k_q_reduction_performed = False
        self.c_reduction_performed = False
        self.rho_reduction_performed = False

        if angles is not None:
            logging.info("Reducing the frame by pre-computing T_K_Q")
            self.reduce_frame_using_t_k_q(angles=angles)
            self.t_k_q_reduction_performed = True

        if magnetic_field_gauss is not None:
            logging.info("Reducing the frame by pre-computing C")
            self.reduce_frame_using_c(magnetic_field_gauss=magnetic_field_gauss)
            self.c_reduction_performed = True

        if rho is not None:
            logging.info("Reducing the frame by pre-computing Rho")
            # Require magnetic field so that the user doesn't use
            # incompatible rho and atmospheric parameters by accident
            assert magnetic_field_gauss is not None, "magnetic_field_gauss must be provided for pre-computing Rho"
            assert self.c_reduction_performed, "C reduction must be performed before Rho reduction"
            self.reduce_frame_using_rho(rho=rho)
            self.rho_reduction_performed = True

        self.frame_s_precomputed = self.frame_s.copy()
        self.frame_s_precomputed_id_columns = self.frame_s_id_columns.copy()

    @log_method
    def reset_frames(self):
        self.frame_s = self.frame_s_precomputed.copy()
        self.frame_s_id_columns = self.frame_s_precomputed_id_columns.copy()

    @log_method
    def precompute_eta_s_frame(self):
        rows = []
        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower

            logging.debug(f"Processing {level_upper.level_id} -> {level_lower.level_id}")
            if self.cutoff_condition(level_upper=level_upper, level_lower=level_lower, nu=self.nu):
                logging.debug(
                    f"Cutting off the transition {level_upper.level_id} -> {level_lower.level_id} "
                    f"because it does not contribute to the specified frequency range"
                )
                continue

            Ll = level_lower.L
            Lu = level_upper.L

            S = level_lower.S
            for ju, Ju, Jʹu, Jʹʹu, jl, Jl, Jʹl, Mu, Mʹu, Ml, K, Ku, Qu, q, qʹ, Q in nested_loops(
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
            ):
                coefficient = multiply(
                    lambda: 1,
                    lambda: n_proj(Lu) * transition.einstein_b_ul * sqrt(n_proj(1, K, Ku)),
                    lambda: m1p(1 + Jʹu - Mu + qʹ),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mʹu, Ml, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹu, Jʹʹu, Ku, Mʹu, -Mu, -Qu),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                )
                rows.append(
                    {
                        "level_upper_id": level_upper.level_id,
                        "level_lower_id": level_lower.level_id,
                        "transition_id": transition.transition_id,
                        "ju": ju,
                        "Ju": Ju,
                        "Jʹu": Jʹu,
                        "Jʹʹu": Jʹʹu,
                        "jl": jl,
                        "Jl": Jl,
                        "Jʹl": Jʹl,
                        "Mu": Mu,
                        "Mʹu": Mʹu,
                        "Ml": Ml,
                        "K": K,
                        "Ku": Ku,
                        "Qu": Qu,
                        "q": q,
                        "qʹ": qʹ,
                        "Q": Q,
                        "coefficient": coefficient,
                    }
                )
        self.frame_s_id_columns = [
            "level_upper_id",
            "level_lower_id",
            "transition_id",
            "ju",
            "Ju",
            "Jʹu",
            "Jʹʹu",
            "jl",
            "Jl",
            "Jʹl",
            "Mu",
            "Mʹu",
            "Ml",
            "K",
            "Ku",
            "Qu",
            "q",
            "qʹ",
            "Q",
        ]
        return pd.DataFrame(rows)

    @log_method
    def reduce_frame_using_t_k_q(self, angles: Angles):
        D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
        D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)

        self.frame_s = self.frame_s.merge(
            self.construct_t_k_q_frame(D_inverse_omega=D_inverse_omega, D_magnetic=D_magnetic),
            on=["K", "Q"],
        )
        self.frame_s["coefficient_t_k_q_I"] = self.frame_s["coefficient"] * self.frame_s["t_k_q_I"]
        self.frame_s["coefficient_t_k_q_Q"] = self.frame_s["coefficient"] * self.frame_s["t_k_q_Q"]
        self.frame_s["coefficient_t_k_q_U"] = self.frame_s["coefficient"] * self.frame_s["t_k_q_U"]
        self.frame_s["coefficient_t_k_q_V"] = self.frame_s["coefficient"] * self.frame_s["t_k_q_V"]

        columns_to_reduce = ["K", "Q", "coefficient", "t_k_q_I", "t_k_q_Q", "t_k_q_U", "t_k_q_V"]
        self.frame_s_id_columns = [col for col in self.frame_s_id_columns if col not in columns_to_reduce]
        self.frame_s = self.frame_s.drop(columns=columns_to_reduce).groupby(self.frame_s_id_columns).sum().reset_index()

    @log_method
    def reduce_frame_using_c(self, magnetic_field_gauss):
        c_frame = self.construct_c_frame(magnetic_field_gauss=magnetic_field_gauss)
        self.frame_s = (
            self.frame_s.merge(
                c_frame.rename(columns={"level_id": "level_lower_id", "j": "jl", "J": "Jl", "M": "Ml", "cpb": "c_1"}),
                on=["level_lower_id", "jl", "Jl", "Ml"],
            )
            .merge(
                c_frame.rename(columns={"level_id": "level_lower_id", "j": "jl", "J": "Jʹl", "M": "Ml", "cpb": "c_2"}),
                on=["level_lower_id", "jl", "Jʹl", "Ml"],
            )
            .merge(
                c_frame.rename(columns={"level_id": "level_upper_id", "j": "ju", "J": "Ju", "M": "Mu", "cpb": "c_3"}),
                on=["level_upper_id", "ju", "Ju", "Mu"],
            )
            .merge(
                c_frame.rename(columns={"level_id": "level_upper_id", "j": "ju", "J": "Jʹʹu", "M": "Mu", "cpb": "c_4"}),
                on=["level_upper_id", "ju", "Jʹʹu", "Mu"],
            )
        )
        self.frame_s["coefficient_t_k_q_I_c"] = np.prod(
            self.frame_s[["coefficient_t_k_q_I", "c_1", "c_2", "c_3", "c_4"]].values, axis=1
        )
        self.frame_s["coefficient_t_k_q_Q_c"] = np.prod(
            self.frame_s[["coefficient_t_k_q_Q", "c_1", "c_2", "c_3", "c_4"]].values, axis=1
        )
        self.frame_s["coefficient_t_k_q_U_c"] = np.prod(
            self.frame_s[["coefficient_t_k_q_U", "c_1", "c_2", "c_3", "c_4"]].values, axis=1
        )
        self.frame_s["coefficient_t_k_q_V_c"] = np.prod(
            self.frame_s[["coefficient_t_k_q_V", "c_1", "c_2", "c_3", "c_4"]].values, axis=1
        )

        columns_to_reduce = [
            "Jl",
            "Jʹl",
            "Ju",
            "coefficient_t_k_q_I",
            "coefficient_t_k_q_Q",
            "coefficient_t_k_q_U",
            "coefficient_t_k_q_V",
            "c_1",
            "c_2",
            "c_3",
            "c_4",
        ]
        self.frame_s_id_columns = [col for col in self.frame_s_id_columns if col not in columns_to_reduce]
        self.frame_s = self.frame_s.drop(columns=columns_to_reduce).groupby(self.frame_s_id_columns).sum().reset_index()

    @log_method
    def reduce_frame_using_rho(self, rho: Rho):
        self.frame_s = self.frame_s.merge(
            self.construct_rho_frame(rho=rho).rename(
                columns={"level_id": "level_upper_id", "K": "Ku", "Q": "Qu", "J": "Jʹu", "Jʹ": "Jʹʹu"}
            ),
            on=["level_upper_id", "Ku", "Qu", "Jʹu", "Jʹʹu"],
        )

        self.frame_s["coefficient_t_k_q_I_c_rho"] = np.prod(
            self.frame_s[["coefficient_t_k_q_I_c", "rho"]].values, axis=1
        )
        self.frame_s["coefficient_t_k_q_Q_c_rho"] = np.prod(
            self.frame_s[["coefficient_t_k_q_Q_c", "rho"]].values, axis=1
        )
        self.frame_s["coefficient_t_k_q_U_c_rho"] = np.prod(
            self.frame_s[["coefficient_t_k_q_U_c", "rho"]].values, axis=1
        )
        self.frame_s["coefficient_t_k_q_V_c_rho"] = np.prod(
            self.frame_s[["coefficient_t_k_q_V_c", "rho"]].values, axis=1
        )

        columns_to_reduce = [
            "Ku",
            "Qu",
            "Jʹu",
            "Jʹʹu",
            "coefficient_t_k_q_I_c",
            "coefficient_t_k_q_Q_c",
            "coefficient_t_k_q_U_c",
            "coefficient_t_k_q_V_c",
            "rho",
        ]  # C should already be merged, so Jʹʹu is no longer needed
        self.frame_s_id_columns = [col for col in self.frame_s_id_columns if col not in columns_to_reduce]
        self.frame_s = self.frame_s.drop(columns=columns_to_reduce).groupby(self.frame_s_id_columns).sum().reset_index()

    @log_method
    def construct_t_k_q_frame(self, D_inverse_omega, D_magnetic):
        rows_tkq = []

        for K, Q in nested_loops(
            K=FROMTO(0, 2),
            Q=PROJECTION("K"),
        ):
            rows_tkq.append(
                {
                    "K": K,
                    "Q": Q,
                    "t_k_q_I": T_K_Q_double_rotation(
                        K=K,
                        Q=Q,
                        stokes_component_index=0,
                        D_inverse_omega=D_inverse_omega,
                        D_magnetic=D_magnetic,
                    ),
                    "t_k_q_Q": T_K_Q_double_rotation(
                        K=K,
                        Q=Q,
                        stokes_component_index=1,
                        D_inverse_omega=D_inverse_omega,
                        D_magnetic=D_magnetic,
                    ),
                    "t_k_q_U": T_K_Q_double_rotation(
                        K=K,
                        Q=Q,
                        stokes_component_index=2,
                        D_inverse_omega=D_inverse_omega,
                        D_magnetic=D_magnetic,
                    ),
                    "t_k_q_V": T_K_Q_double_rotation(
                        K=K,
                        Q=Q,
                        stokes_component_index=3,
                        D_inverse_omega=D_inverse_omega,
                        D_magnetic=D_magnetic,
                    ),
                }
            )
        return pd.DataFrame(rows_tkq)

    @log_method
    def construct_rho_frame(self, rho: Rho):
        rows_rho = []
        for level in self.term_registry.levels.values():
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(level.L, level.S),
                Jʹ=TRIANGULAR(level.L, level.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=INTERSECTION(PROJECTION("K")),
            ):
                rows_rho.append(
                    {
                        "level_id": level.level_id,
                        "J": J,
                        "Jʹ": Jʹ,
                        "K": K,
                        "Q": Q,
                        "rho": rho(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ),
                    }
                )
        return pd.DataFrame(rows_rho)

    @log_method
    def construct_c_frame(self, magnetic_field_gauss):
        rows_c = []
        for level in self.term_registry.levels.values():
            pb_eigenvalues, pb_eigenvectors = calculate_paschen_back(
                level=level, magnetic_field_gauss=magnetic_field_gauss
            )

            for j, J, M in nested_loops(
                j=TRIANGULAR(level.L, level.S),
                J=TRIANGULAR(level.L, level.S),
                M=INTERSECTION(PROJECTION("J"), PROJECTION("j")),
            ):
                rows_c.append(
                    {
                        "level_id": level.level_id,
                        "j": j,
                        "J": J,
                        "M": M,
                        "cpb": pb_eigenvectors(j=j, J=J, level=level, M=M),
                    }
                )

        return pd.DataFrame(rows_c)

    @log_method
    def construct_phi_frame(self, atmosphere_parameters: AtmosphereParameters):
        rows_phi = []
        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower

            if self.cutoff_condition(level_upper=level_upper, level_lower=level_lower, nu=self.nu):
                continue

            Ll = level_lower.L
            Lu = level_upper.L

            S = level_lower.S
            lower_pb_eigenvalues, lower_pb_eigenvectors = calculate_paschen_back(
                level=level_lower, magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss
            )
            upper_pb_eigenvalues, upper_pb_eigenvectors = calculate_paschen_back(
                level=level_upper, magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss
            )

            for ju, jl, Mu, Ml in nested_loops(
                ju=TRIANGULAR(Lu, S),
                jl=TRIANGULAR(Ll, S),
                Mu=INTERSECTION(PROJECTION("ju")),
                Ml=INTERSECTION(PROJECTION("jl")),
            ):
                rows_phi.append(
                    {
                        "level_upper_id": level_upper.level_id,
                        "level_lower_id": level_lower.level_id,
                        "transition_id": transition.transition_id,
                        "ju": ju,
                        "jl": jl,
                        "Mu": Mu,
                        "Ml": Ml,
                        "phi": self.phi(
                            nui=energy_cmm1_to_frequency_hz(
                                upper_pb_eigenvalues(j=ju, level=level_upper, M=Mu)
                                - lower_pb_eigenvalues(j=jl, level=level_lower, M=Ml),
                            ),
                            nu=self.nu,
                            macroscopic_velocity_cm_sm1=atmosphere_parameters.macroscopic_velocity_cm_sm1,
                            delta_v_thermal_cm_sm1=atmosphere_parameters.delta_v_thermal_cm_sm1,
                            voigt_a=atmosphere_parameters.voigt_a,
                        ),
                    }
                )

        return pd.DataFrame(rows_phi)

    @log_method
    def eta_rho_s(self, atmosphere_parameters: AtmosphereParameters, angles: Angles = None, rho: Rho = None):
        """
        Reference:
        (7.47b)
        """
        self.reset_frames()

        if not self.t_k_q_reduction_performed:
            assert angles is not None, "Angles should be provided if T_K_Q reduction is not performed"
            self.reduce_frame_using_t_k_q(angles=angles)
        else:
            assert angles is None, "Angles should not be provided if T_K_Q reduction is already performed"

        if not self.c_reduction_performed:
            self.reduce_frame_using_c(magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss)
        else:
            assert (
                atmosphere_parameters.magnetic_field_gauss == self.magnetic_field_gauss
            ), "Atmosphere parameters magnetic field gauss should be the same as the one used for C reduction"

        if not self.rho_reduction_performed:
            assert rho is not None, "Rho should be provided if Rho reduction is not performed"
            self.reduce_frame_using_rho(rho=rho)
        else:
            assert rho is None, "Rho should not be provided if Rho reduction is already performed"

        phi_frame = self.construct_phi_frame(atmosphere_parameters=atmosphere_parameters)
        final_frame = self.frame_s.merge(
            phi_frame,
            on=[
                "level_upper_id",
                "level_lower_id",
                "transition_id",
                "ju",
                "jl",
                "Mu",
                "Ml",
            ],
        )

        results = [
            np.prod(
                final_frame[[t_k_q, "phi"]].values,
                axis=1,
            ).sum()
            for t_k_q in [
                "coefficient_t_k_q_I_c_rho",
                "coefficient_t_k_q_Q_c_rho",
                "coefficient_t_k_q_U_c_rho",
                "coefficient_t_k_q_V_c_rho",
            ]
        ]

        result_I, result_Q, result_U, result_V = [h_erg_s * self.nu / 4 / pi * self.N * result for result in results]

        return result_I, result_Q, result_U, result_V

    def phi(self, nui, nu, macroscopic_velocity_cm_sm1, delta_v_thermal_cm_sm1, voigt_a):
        """
        Reference: (5.43 - 5.45)
        """
        delta_nu_D = nui * delta_v_thermal_cm_sm1 / c_cm_sm1  # Doppler width

        nu_round = (nui - nu) / delta_nu_D  # nui already accounts for magnetic shifts
        nu_round_A = macroscopic_velocity_cm_sm1 / delta_v_thermal_cm_sm1

        complex_voigt = voigt(nu=nu_round - nu_round_A, a=voigt_a) / sqrt_pi / delta_nu_D
        return complex_voigt

    def cutoff_condition(self, level_upper: Level, level_lower: Level, nu: np.ndarray):
        nui = energy_cmm1_to_frequency_hz(level_upper.get_mean_energy_cmm1() - level_lower.get_mean_energy_cmm1())
        cutoff = self.delta_nu_cutoff
        if min(nu) > nui + cutoff or max(nu) < nui - cutoff:
            return True
        return False

    @staticmethod
    def epsilon(eta_s: np.ndarray, nu: np.ndarray):
        """
        Reference:
        (7.47e)
        """
        return 2 * h_erg_s * nu**3 / c_cm_sm1**2 * np.real(eta_s)
