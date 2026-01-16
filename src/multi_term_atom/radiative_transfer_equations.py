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
from src.engine.functions.looping import (
    FROMTO,
    INTERSECTION,
    PROJECTION,
    TRIANGULAR,
    VALUE,
)
from src.engine.generators.merge_frame import Frame, SumLimits
from src.engine.generators.merge_loopers import (
    DummyOrAlreadyMerged,
    FromTo,
    Intersection,
    Projection,
    Triangular,
    Value,
    vector,
    Constraint,
    ApplyConstraint,
)
from src.engine.generators.multiply import multiply
from src.engine.generators.nested_loops import nested_loops
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
    """
    level_registry: defines all levels within the atom model.
    transition_registry: defines all transitions within the atom model.
    nu: defines the spectral interval of interest for cutting off non-contributing transitions.
    maximum_delta_v_thermal_units_cutoff: defines the cutoff condition for the transition.
    angles: optional, if provided outright, allows to reduce the frame by precomputing T_K_Q tensor
    rho: optional, if provided outright, allows to reduce the frame by precomputing rho matrix
    N: atom concentration
    precompute: set to False if only the new Frame engine will be used.
    j_constrained: set to True and provide specific J values in a transition to limit it only to certain J.
    """

    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        nu: np.ndarray,
        delta_nu_cutoff=None,
        angles: Angles = None,
        magnetic_field_gauss=None,
        rho: Rho = None,
        N=1,
        precompute=True,
            j_constrained=False,

    ):
        self.level_registry: LevelRegistry = level_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.nu = nu
        self.delta_nu_cutoff = delta_nu_cutoff
        if self.delta_nu_cutoff is None:
            self.delta_nu_cutoff = max(10 * (np.max(nu) - np.min(nu)), np.mean(nu) * 1e-3)
        self.N = N  # Atom concentration
        self.precompute = precompute
        self.j_constrained=j_constrained
        self.frame_a_id_columns = []
        self.frame_s_id_columns = []
        if self.precompute:
            self.frame_a = self.precompute_a_frame()
            self.frame_s = self.precompute_s_frame()
        self.magnetic_field_gauss = magnetic_field_gauss
        self.t_k_q_reduction_performed = False
        self.c_reduction_performed = False
        self.rho_reduction_performed = False

        if angles is not None and self.precompute:
            logging.info("Reducing the frame by pre-computing T_K_Q")
            self.reduce_a_frame_using_t_k_q(angles=angles)
            self.reduce_s_frame_using_t_k_q(angles=angles)
            self.t_k_q_reduction_performed = True

        if magnetic_field_gauss is not None and self.precompute:
            logging.info("Reducing the frame by pre-computing C")
            assert angles is not None, "angles must be provided for pre-computing C"
            self.reduce_a_frame_using_c(magnetic_field_gauss=magnetic_field_gauss)
            self.reduce_s_frame_using_c(magnetic_field_gauss=magnetic_field_gauss)
            self.c_reduction_performed = True

        if rho is not None and self.precompute:
            logging.info("Reducing the frame by pre-computing Rho")
            # Require magnetic field so that the user doesn't use
            # incompatible rho and atmospheric parameters by accident
            assert magnetic_field_gauss is not None, "magnetic_field_gauss must be provided for pre-computing Rho"
            assert self.c_reduction_performed, "C reduction must be performed before Rho reduction"
            self.reduce_a_frame_using_rho(rho=rho)
            self.reduce_s_frame_using_rho(rho=rho)
            self.rho_reduction_performed = True

        if self.precompute:
            self.frame_a_precomputed = self.frame_a.copy()
            self.frame_s_precomputed = self.frame_s.copy()
            self.frame_a_precomputed_id_columns = self.frame_a_id_columns.copy()
            self.frame_s_precomputed_id_columns = self.frame_s_id_columns.copy()

    @log_method
    def reset_frames(self):
        self.frame_a = self.frame_a_precomputed.copy()
        self.frame_s = self.frame_s_precomputed.copy()
        self.frame_a_id_columns = self.frame_a_precomputed_id_columns.copy()
        self.frame_s_id_columns = self.frame_s_precomputed_id_columns.copy()

    @log_method
    def precompute_a_frame(self):
        rows = []
        for transition in self.transition_registry.transitions.values():
            term_upper = transition.term_upper
            term_lower = transition.term_lower

            logging.debug(f"Processing {term_upper.term_id} -> {term_lower.term_id}")
            if self.cutoff_condition(term_upper=term_upper, term_lower=term_lower, nu=self.nu):
                logging.debug(
                    f"Cutting off the transition {term_upper.term_id} -> {term_lower.term_id} "
                    f"because it does not contribute to the specified frequency range"
                )
                continue

            Ll = term_lower.L
            Lu = term_upper.L

            S = term_lower.S
            for jl, Jl, Jʹl, Jʹʹl, ju, Ju, Jʹu, Ml, Mʹl, Mu, K, Kl, Ql, q, qʹ, Q in nested_loops(
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
            ):
                coefficient = multiply(
                    lambda: n_proj(Ll),
                    lambda: transition.einstein_b_lu * sqrt(n_proj(1, K, Kl)),
                    lambda: m1p(1 + Jʹʹl - Ml + qʹ),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mu, Mʹl, -qʹ),
                    lambda: wigner_3j(1, 1, K, q, -qʹ, -Q),
                    lambda: wigner_3j(Jʹʹl, Jʹl, Kl, Ml, -Mʹl, -Ql),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                )
                rows.append(
                    {
                        "term_upper_id": term_upper.term_id,
                        "term_lower_id": term_lower.term_id,
                        "transition_id": transition.transition_id,
                        "jl": jl,
                        "Jl": Jl,
                        "Jʹl": Jʹl,
                        "Jʹʹl": Jʹʹl,
                        "ju": ju,
                        "Ju": Ju,
                        "Jʹu": Jʹu,
                        "Ml": Ml,
                        "Mʹl": Mʹl,
                        "Mu": Mu,
                        "K": K,
                        "Kl": Kl,
                        "Ql": Ql,
                        "q": q,
                        "qʹ": qʹ,
                        "Q": Q,
                        "coefficient": coefficient,
                    }
                )
        self.frame_a_id_columns = [
            "term_upper_id",
            "term_lower_id",
            "transition_id",
            "jl",
            "Jl",
            "Jʹl",
            "Jʹʹl",
            "ju",
            "Ju",
            "Jʹu",
            "Ml",
            "Mʹl",
            "Mu",
            "K",
            "Kl",
            "Ql",
            "q",
            "qʹ",
            "Q",
        ]
        return pd.DataFrame(rows)

    class AFrameSumLimits(SumLimits):
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

    @log_method
    def frame_a_frame(
        self,
        stokes_component_index: int,
        angles: Angles,
        magnetic_field_gauss: float,
        rho: Rho,
        atmosphere_parameters: AtmosphereParameters,
    ):
        sum_limits = self.AFrameSumLimitsConstrained() if self.j_constrained else self.AFrameSumLimits()

        frame = Frame.from_sum_limits(
            base_frame=self.get_base_frame(),
            sum_linits=sum_limits,
        )

        frame.add_factors_to_multiply(
            n_proj=lambda Ll:                              n_proj(Ll),
            b_lu=lambda einstein_b_lu, K, Kl:              einstein_b_lu * sqrt(n_proj(1, K, Kl)),
            m1p=lambda Jʹʹl, Ml, qʹ:                       m1p(1 + Jʹʹl - Ml + qʹ),
            sqrt=lambda Jl, Jʹl, Ju, Jʹu:                  sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
            wigner_3j1=lambda Ju, Jl, Mu, Ml, q:           wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
            wigner_3j2=lambda Jʹu, Jʹl, Mu, Mʹl, qʹ:       wigner_3j(Jʹu, Jʹl, 1, -Mu, Mʹl, -qʹ),
            wigner_3j3=lambda K, q, qʹ, Q:                 wigner_3j(1, 1, K, q, -qʹ, -Q),
            wigner_3j4=lambda Jʹʹl, Jʹl, Kl, Ml, Mʹl, Ql:  wigner_3j(Jʹʹl, Jʹl, Kl, Ml, -Mʹl, -Ql),
            wigner_6j1=lambda Lu, Ll, Jl, Ju, S:           wigner_6j(Lu, Ll, 1, Jl, Ju, S),
            wigner_6j2=lambda Lu, Ll, Jʹl, Jʹu, S:         wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
        )  # fmt: skip

        D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
        D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)
        frame.add_factors_to_multiply(
            tkq=lambda K, Q: T_K_Q_double_rotation(
                K=K,
                Q=Q,
                stokes_component_index=stokes_component_index,
                D_inverse_omega=D_inverse_omega,
                D_magnetic=D_magnetic,
            ),
            elementwise=True,
        )

        precalculated_pb_eigenvalues = {}
        precalculated_pb_eigenvectors = {}
        for term in self.level_registry.terms.values():
            pb_eigenvalues, pb_eigenvectors = calculate_paschen_back(
                term=term, magnetic_field_gauss=magnetic_field_gauss
            )
            precalculated_pb_eigenvalues[term.term_id] = pb_eigenvalues
            precalculated_pb_eigenvectors[term.term_id] = pb_eigenvectors

        frame.add_factors_to_multiply(
            pb1=lambda term_lower_id, jl, Jl, Ml: precalculated_pb_eigenvectors[term_lower_id](j=jl, J=Jl, M=Ml),
            pb2=lambda term_lower_id, jl, Jʹʹl, Ml: precalculated_pb_eigenvectors[term_lower_id](j=jl, J=Jʹʹl, M=Ml),
            pb3=lambda term_upper_id, ju, Ju, Mu: precalculated_pb_eigenvectors[term_upper_id](j=ju, J=Ju, M=Mu),
            pb4=lambda term_upper_id, ju, Jʹu, Mu: precalculated_pb_eigenvectors[term_upper_id](j=ju, J=Jʹu, M=Mu),
            elementwise=True,
        )

        frame.add_factors_to_multiply(
            rho=lambda term_lower_id, Kl, Ql, Jʹʹl, Jʹl: rho(term_id=term_lower_id, K=Kl, Q=Ql, J=Jʹʹl, Jʹ=Jʹl),
            elementwise=True,
        )

        frame.add_factors_to_multiply(
            phi=lambda ju, Mu, term_upper_id, jl, Ml, term_lower_id: (
                self.phi(
                    nui=energy_cmm1_to_frequency_hz(
                        precalculated_pb_eigenvalues[term_upper_id](j=ju, M=Mu)
                        - precalculated_pb_eigenvalues[term_lower_id](j=jl, M=Ml)
                    ),
                    nu=self.nu,
                    macroscopic_velocity_cm_sm1=atmosphere_parameters.macroscopic_velocity_cm_sm1,
                    delta_v_thermal_cm_sm1=atmosphere_parameters.delta_v_thermal_cm_sm1,
                    voigt_a=atmosphere_parameters.voigt_a,
                )
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
            ...,
        )
        result = h_erg_s * self.nu / 4 / pi * self.N * result
        return result

    class SFrameSumLimits(SumLimits):
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
    @log_method
    def frame_s_frame(
        self,
        stokes_component_index: int,
        angles: Angles,
        magnetic_field_gauss: float,
        rho: Rho,
        atmosphere_parameters: AtmosphereParameters,
    ):
        sum_limits = self.SFrameSumLimitsConstrained() if self.j_constrained else self.SFrameSumLimits()

        frame = Frame.from_sum_limits(
            base_frame=self.get_base_frame(),
            sum_linits=sum_limits,
        )

        frame.add_factors_to_multiply(
            n_proj=lambda Lu: n_proj(Lu),
            b_ul=lambda einstein_b_ul, K, Ku: einstein_b_ul * sqrt(n_proj(1, K, Ku)),
            m1p=lambda Jʹu, Mu, qʹ: m1p(1 + Jʹu - Mu + qʹ),
            sqrt=lambda Jl, Jʹl, Ju, Jʹu: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
            wigner_3j1=lambda Ju, Jl, Mu, Ml, q: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
            wigner_3j2=lambda Jʹu, Jʹl, Mʹu, Ml, qʹ: wigner_3j(Jʹu, Jʹl, 1, -Mʹu, Ml, -qʹ),
            wigner_3j3=lambda K, q, qʹ, Q: wigner_3j(1, 1, K, q, -qʹ, -Q),
            wigner_3j4=lambda Jʹʹu, Jʹu, Ku, Mu, Mʹu, Qu: wigner_3j(Jʹu, Jʹʹu, Ku, Mʹu, -Mu, -Qu),
            wigner_6j1=lambda Lu, Ll, Jl, Ju, S: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
            wigner_6j2=lambda Lu, Ll, Jʹl, Jʹu, S: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
        )  # fmt: skip

        D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
        D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)
        frame.add_factors_to_multiply(
            tkq=lambda K, Q: T_K_Q_double_rotation(
                K=K,
                Q=Q,
                stokes_component_index=stokes_component_index,
                D_inverse_omega=D_inverse_omega,
                D_magnetic=D_magnetic,
            ),
            elementwise=True,
        )

        precalculated_pb_eigenvalues = {}
        precalculated_pb_eigenvectors = {}
        for term in self.level_registry.terms.values():
            pb_eigenvalues, pb_eigenvectors = calculate_paschen_back(
                term=term, magnetic_field_gauss=magnetic_field_gauss
            )
            precalculated_pb_eigenvalues[term.term_id] = pb_eigenvalues
            precalculated_pb_eigenvectors[term.term_id] = pb_eigenvectors

        frame.add_factors_to_multiply(
            pb1=lambda term_lower_id, jl, Jl, Ml: precalculated_pb_eigenvectors[term_lower_id](j=jl, J=Jl, M=Ml),
            pb2=lambda term_lower_id, jl, Jʹl, Ml: precalculated_pb_eigenvectors[term_lower_id](j=jl, J=Jʹl, M=Ml),
            pb3=lambda term_upper_id, ju, Ju, Mu: precalculated_pb_eigenvectors[term_upper_id](j=ju, J=Ju, M=Mu),
            pb4=lambda term_upper_id, ju, Jʹʹu, Mu: precalculated_pb_eigenvectors[term_upper_id](j=ju, J=Jʹʹu, M=Mu),
            elementwise=True,
        )

        frame.add_factors_to_multiply(
            rho=lambda term_upper_id, Ku, Qu, Jʹʹu, Jʹu: rho(term_id=term_upper_id, K=Ku, Q=Qu, J=Jʹu, Jʹ=Jʹʹu),
            elementwise=True,
        )

        frame.add_factors_to_multiply(
            phi=lambda ju, Mu, term_upper_id, jl, Ml, term_lower_id: (
                self.phi(
                    nui=energy_cmm1_to_frequency_hz(
                        precalculated_pb_eigenvalues[term_upper_id](j=ju, M=Mu)
                        - precalculated_pb_eigenvalues[term_lower_id](j=jl, M=Ml)
                    ),
                    nu=self.nu,
                    macroscopic_velocity_cm_sm1=atmosphere_parameters.macroscopic_velocity_cm_sm1,
                    delta_v_thermal_cm_sm1=atmosphere_parameters.delta_v_thermal_cm_sm1,
                    voigt_a=atmosphere_parameters.voigt_a,
                )
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
            ...,
        )
        result = h_erg_s * self.nu / 4 / pi * self.N * result
        return result

    def get_base_frame(self):
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
    def precompute_s_frame(self):
        rows = []
        for transition in self.transition_registry.transitions.values():
            term_upper = transition.term_upper
            term_lower = transition.term_lower

            logging.debug(f"Processing {term_upper.term_id} -> {term_lower.term_id}")
            if self.cutoff_condition(term_upper=term_upper, term_lower=term_lower, nu=self.nu):
                logging.debug(
                    f"Cutting off the transition {term_upper.term_id} -> {term_lower.term_id} "
                    f"because it does not contribute to the specified frequency range"
                )
                continue

            Ll = term_lower.L
            Lu = term_upper.L

            S = term_lower.S
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
                        "term_upper_id": term_upper.term_id,
                        "term_lower_id": term_lower.term_id,
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
            "term_upper_id",
            "term_lower_id",
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
    def reduce_a_frame_using_t_k_q(self, angles: Angles):
        D_inverse_omega = WignerD(alpha=-angles.gamma, beta=-angles.theta, gamma=-angles.chi, K_max=2)
        D_magnetic = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)

        self.frame_a = self.frame_a.merge(
            self.construct_t_k_q_frame(D_inverse_omega=D_inverse_omega, D_magnetic=D_magnetic),
            on=["K", "Q"],
        )
        self.frame_a["coefficient_t_k_q_I"] = self.frame_a["coefficient"] * self.frame_a["t_k_q_I"]
        self.frame_a["coefficient_t_k_q_Q"] = self.frame_a["coefficient"] * self.frame_a["t_k_q_Q"]
        self.frame_a["coefficient_t_k_q_U"] = self.frame_a["coefficient"] * self.frame_a["t_k_q_U"]
        self.frame_a["coefficient_t_k_q_V"] = self.frame_a["coefficient"] * self.frame_a["t_k_q_V"]

        columns_to_reduce = ["K", "Q", "coefficient", "t_k_q_I", "t_k_q_Q", "t_k_q_U", "t_k_q_V"]
        self.frame_a_id_columns = [col for col in self.frame_a_id_columns if col not in columns_to_reduce]
        self.frame_a = self.frame_a.drop(columns=columns_to_reduce).groupby(self.frame_a_id_columns).sum().reset_index()

    @log_method
    def reduce_s_frame_using_t_k_q(self, angles: Angles):
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
    def reduce_a_frame_using_c(self, magnetic_field_gauss):
        c_frame = self.construct_c_frame(magnetic_field_gauss=magnetic_field_gauss)
        self.frame_a = (
            self.frame_a.merge(
                c_frame.rename(columns={"term_id": "term_lower_id", "j": "jl", "J": "Jl", "M": "Ml", "cpb": "c_1"}),
                on=["term_lower_id", "jl", "Jl", "Ml"],
            )
            .merge(
                c_frame.rename(columns={"term_id": "term_lower_id", "j": "jl", "J": "Jʹʹl", "M": "Ml", "cpb": "c_2"}),
                on=["term_lower_id", "jl", "Jʹʹl", "Ml"],
            )
            .merge(
                c_frame.rename(columns={"term_id": "term_upper_id", "j": "ju", "J": "Ju", "M": "Mu", "cpb": "c_3"}),
                on=["term_upper_id", "ju", "Ju", "Mu"],
            )
            .merge(
                c_frame.rename(columns={"term_id": "term_upper_id", "j": "ju", "J": "Jʹu", "M": "Mu", "cpb": "c_4"}),
                on=["term_upper_id", "ju", "Jʹu", "Mu"],
            )
        )
        self.frame_a["coefficient_t_k_q_I_c"] = np.prod(
            self.frame_a[["coefficient_t_k_q_I", "c_1", "c_2", "c_3", "c_4"]].values, axis=1
        )
        self.frame_a["coefficient_t_k_q_Q_c"] = np.prod(
            self.frame_a[["coefficient_t_k_q_Q", "c_1", "c_2", "c_3", "c_4"]].values, axis=1
        )
        self.frame_a["coefficient_t_k_q_U_c"] = np.prod(
            self.frame_a[["coefficient_t_k_q_U", "c_1", "c_2", "c_3", "c_4"]].values, axis=1
        )
        self.frame_a["coefficient_t_k_q_V_c"] = np.prod(
            self.frame_a[["coefficient_t_k_q_V", "c_1", "c_2", "c_3", "c_4"]].values, axis=1
        )

        columns_to_reduce = [
            "Ju",
            "Jʹu",
            "Jl",
            "coefficient_t_k_q_I",
            "coefficient_t_k_q_Q",
            "coefficient_t_k_q_U",
            "coefficient_t_k_q_V",
            "c_1",
            "c_2",
            "c_3",
            "c_4",
        ]
        self.frame_a_id_columns = [col for col in self.frame_a_id_columns if col not in columns_to_reduce]
        self.frame_a = self.frame_a.drop(columns=columns_to_reduce).groupby(self.frame_a_id_columns).sum().reset_index()

    @log_method
    def reduce_s_frame_using_c(self, magnetic_field_gauss):
        c_frame = self.construct_c_frame(magnetic_field_gauss=magnetic_field_gauss)
        self.frame_s = (
            self.frame_s.merge(
                c_frame.rename(columns={"term_id": "term_lower_id", "j": "jl", "J": "Jl", "M": "Ml", "cpb": "c_1"}),
                on=["term_lower_id", "jl", "Jl", "Ml"],
            )
            .merge(
                c_frame.rename(columns={"term_id": "term_lower_id", "j": "jl", "J": "Jʹl", "M": "Ml", "cpb": "c_2"}),
                on=["term_lower_id", "jl", "Jʹl", "Ml"],
            )
            .merge(
                c_frame.rename(columns={"term_id": "term_upper_id", "j": "ju", "J": "Ju", "M": "Mu", "cpb": "c_3"}),
                on=["term_upper_id", "ju", "Ju", "Mu"],
            )
            .merge(
                c_frame.rename(columns={"term_id": "term_upper_id", "j": "ju", "J": "Jʹʹu", "M": "Mu", "cpb": "c_4"}),
                on=["term_upper_id", "ju", "Jʹʹu", "Mu"],
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
    def reduce_a_frame_using_rho(self, rho: Rho):
        self.frame_a = self.frame_a.merge(
            self.construct_rho_frame(rho=rho).rename(
                columns={"term_id": "term_lower_id", "K": "Kl", "Q": "Ql", "J": "Jʹʹl", "Jʹ": "Jʹl"}
            ),
            on=["term_lower_id", "Kl", "Ql", "Jʹʹl", "Jʹl"],
        )

        self.frame_a["coefficient_t_k_q_I_c_rho"] = np.prod(
            self.frame_a[["coefficient_t_k_q_I_c", "rho"]].values, axis=1
        )
        self.frame_a["coefficient_t_k_q_Q_c_rho"] = np.prod(
            self.frame_a[["coefficient_t_k_q_Q_c", "rho"]].values, axis=1
        )
        self.frame_a["coefficient_t_k_q_U_c_rho"] = np.prod(
            self.frame_a[["coefficient_t_k_q_U_c", "rho"]].values, axis=1
        )
        self.frame_a["coefficient_t_k_q_V_c_rho"] = np.prod(
            self.frame_a[["coefficient_t_k_q_V_c", "rho"]].values, axis=1
        )

        columns_to_reduce = [
            "Kl",
            "Ql",
            "Jʹʹl",
            "Jʹl",
            "coefficient_t_k_q_I_c",
            "coefficient_t_k_q_Q_c",
            "coefficient_t_k_q_U_c",
            "coefficient_t_k_q_V_c",
            "rho",
        ]  # C should already be merged, so Jʹʹl is no longer needed
        self.frame_a_id_columns = [col for col in self.frame_a_id_columns if col not in columns_to_reduce]
        self.frame_a = self.frame_a.drop(columns=columns_to_reduce).groupby(self.frame_a_id_columns).sum().reset_index()

    @log_method
    def reduce_s_frame_using_rho(self, rho: Rho):
        self.frame_s = self.frame_s.merge(
            self.construct_rho_frame(rho=rho).rename(
                columns={"term_id": "term_upper_id", "K": "Ku", "Q": "Qu", "J": "Jʹu", "Jʹ": "Jʹʹu"}
            ),
            on=["term_upper_id", "Ku", "Qu", "Jʹu", "Jʹʹu"],
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
        for term in self.level_registry.terms.values():
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=INTERSECTION(PROJECTION("K")),
            ):
                rows_rho.append(
                    {
                        "term_id": term.term_id,
                        "J": J,
                        "Jʹ": Jʹ,
                        "K": K,
                        "Q": Q,
                        "rho": rho(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ),
                    }
                )
        return pd.DataFrame(rows_rho)

    @log_method
    def construct_c_frame(self, magnetic_field_gauss):
        rows_c = []
        for term in self.level_registry.terms.values():
            pb_eigenvalues, pb_eigenvectors = calculate_paschen_back(
                term=term, magnetic_field_gauss=magnetic_field_gauss
            )

            for j, J, M in nested_loops(
                j=TRIANGULAR(term.L, term.S),
                J=TRIANGULAR(term.L, term.S),
                M=INTERSECTION(PROJECTION("J"), PROJECTION("j")),
            ):
                rows_c.append(
                    {
                        "term_id": term.term_id,
                        "j": j,
                        "J": J,
                        "M": M,
                        "cpb": pb_eigenvectors(j=j, J=J, term=term, M=M),
                    }
                )

        return pd.DataFrame(rows_c)

    @log_method
    def construct_phi_frame(self, atmosphere_parameters: AtmosphereParameters):
        rows_phi = []
        for transition in self.transition_registry.transitions.values():
            term_upper = transition.term_upper
            term_lower = transition.term_lower

            if self.cutoff_condition(term_upper=term_upper, term_lower=term_lower, nu=self.nu):
                continue

            Ll = term_lower.L
            Lu = term_upper.L

            S = term_lower.S
            lower_pb_eigenvalues, lower_pb_eigenvectors = calculate_paschen_back(
                term=term_lower, magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss
            )
            upper_pb_eigenvalues, upper_pb_eigenvectors = calculate_paschen_back(
                term=term_upper, magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss
            )

            for ju, jl, Mu, Ml in nested_loops(
                ju=TRIANGULAR(Lu, S),
                jl=TRIANGULAR(Ll, S),
                Mu=INTERSECTION(PROJECTION("ju")),
                Ml=INTERSECTION(PROJECTION("jl")),
            ):
                rows_phi.append(
                    {
                        "term_upper_id": term_upper.term_id,
                        "term_lower_id": term_lower.term_id,
                        "transition_id": transition.transition_id,
                        "ju": ju,
                        "jl": jl,
                        "Mu": Mu,
                        "Ml": Ml,
                        "phi": self.phi(
                            nui=energy_cmm1_to_frequency_hz(
                                upper_pb_eigenvalues(j=ju, term=term_upper, M=Mu)
                                - lower_pb_eigenvalues(j=jl, term=term_lower, M=Ml),
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
    def eta_rho_a(self, atmosphere_parameters: AtmosphereParameters, angles: Angles = None, rho: Rho = None):
        """
        Reference:
        (7.47b)
        """
        self.reset_frames()

        if not self.t_k_q_reduction_performed:
            assert angles is not None, "Angles should be provided if T_K_Q reduction is not performed"
            self.reduce_a_frame_using_t_k_q(angles=angles)
        else:
            assert angles is None, "Angles should not be provided if T_K_Q reduction is already performed"

        if not self.c_reduction_performed:
            self.reduce_a_frame_using_c(magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss)
        else:
            assert (
                atmosphere_parameters.magnetic_field_gauss == self.magnetic_field_gauss
            ), "Atmosphere parameters magnetic field gauss should be the same as the one used for C reduction"

        if not self.rho_reduction_performed:
            assert rho is not None, "Rho should be provided if Rho reduction is not performed"
            self.reduce_a_frame_using_rho(rho=rho)
        else:
            assert rho is None, "Rho should not be provided if Rho reduction is already performed"

        phi_frame = self.construct_phi_frame(atmosphere_parameters=atmosphere_parameters)
        final_frame = self.frame_a.merge(
            phi_frame,
            on=[
                "term_upper_id",
                "term_lower_id",
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

    @log_method
    def eta_rho_s(self, atmosphere_parameters: AtmosphereParameters, angles: Angles = None, rho: Rho = None):
        """
        Reference:
        (7.47b)
        """
        self.reset_frames()

        if not self.t_k_q_reduction_performed:
            assert angles is not None, "Angles should be provided if T_K_Q reduction is not performed"
            self.reduce_s_frame_using_t_k_q(angles=angles)
        else:
            assert angles is None, "Angles should not be provided if T_K_Q reduction is already performed"

        if not self.c_reduction_performed:
            self.reduce_s_frame_using_c(magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss)
        else:
            assert (
                atmosphere_parameters.magnetic_field_gauss == self.magnetic_field_gauss
            ), "Atmosphere parameters magnetic field gauss should be the same as the one used for C reduction"

        if not self.rho_reduction_performed:
            assert rho is not None, "Rho should be provided if Rho reduction is not performed"
            self.reduce_s_frame_using_rho(rho=rho)
        else:
            assert rho is None, "Rho should not be provided if Rho reduction is already performed"

        phi_frame = self.construct_phi_frame(atmosphere_parameters=atmosphere_parameters)
        final_frame = self.frame_s.merge(
            phi_frame,
            on=[
                "term_upper_id",
                "term_lower_id",
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

    @log_method
    def compute_all_coefficients(
        self, atmosphere_parameters: AtmosphereParameters, angles: Angles = None, rho: Rho = None
    ):
        eta_rho_aI, eta_rho_aQ, eta_rho_aU, eta_rho_aV = self.eta_rho_a(
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
            rho=rho,
        )

        eta_rho_sI, eta_rho_sQ, eta_rho_sU, eta_rho_sV = self.eta_rho_s(
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
            rho=rho,
        )

        epsilonI = self.epsilon(eta_s=eta_rho_sI, nu=self.nu)
        epsilonQ = self.epsilon(eta_s=eta_rho_sQ, nu=self.nu)
        epsilonU = self.epsilon(eta_s=eta_rho_sU, nu=self.nu)
        epsilonV = self.epsilon(eta_s=eta_rho_sV, nu=self.nu)

        #
        # for transition in self.transition_registry.transitions.values():
        #     term_upper = transition.term_upper
        #     term_lower = transition.term_lower
        #     Lu = term_upper.L
        #     Ll = term_lower.L
        #     S = term_lower.S
        #     epsilon_prime = 1
        #
        #     k_A_M = h_erg_s * self.nu / 4 / pi * self.N * transition.einstein_b_lu
        #
        #     epsilonI += 100*summate(
        #         lambda Ju, Jl: epsilon_prime
        #         / (1 + epsilon_prime)
        #         * k_A_M
        #         * get_planck_BP(nu_sm1= energy_cmm1_to_frequency_hz(term_upper.get_level(Ju).energy_cmm1 - term_lower.get_level(Jl).energy_cmm1), T_K=1000000)
        #         * n_proj(Ju, Jl)
        #         / n_proj(S)
        #         * (wigner_6j(Lu, Ll, 1, Jl, Ju, S)) ** 2
        #         * np.real(
        #             self.phi(
        #                 nui= energy_cmm1_to_frequency_hz(term_upper.get_level(Ju).energy_cmm1 - term_lower.get_level(Jl).energy_cmm1),
        #                 nu=self.nu,
        #                 macroscopic_velocity_cm_sm1=atmosphere_parameters.macroscopic_velocity_cm_sm1,
        #                 delta_v_thermal_cm_sm1=atmosphere_parameters.delta_v_thermal_cm_sm1/3,
        #                 voigt_a=atmosphere_parameters.voigt_a,
        #             )
        #         ),
        #         Ju=TRIANGULAR(Lu, S),
        #         Jl=TRIANGULAR(Ll, S),
        #     )
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


    @log_method
    def compute_all_coefficients_frame(
        self, atmosphere_parameters: AtmosphereParameters, angles: Angles = None, rho: Rho = None
    ):
        eta_rho_aI = self.frame_a_frame(
            stokes_component_index=0,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_aQ = self.frame_a_frame(
            stokes_component_index=1,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_aU = self.frame_a_frame(
            stokes_component_index=2,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_aV = self.frame_a_frame(
            stokes_component_index=3,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_sI = self.frame_s_frame(
            stokes_component_index=0,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_sQ = self.frame_s_frame(
            stokes_component_index=1,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_sU = self.frame_s_frame(
            stokes_component_index=2,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )
        eta_rho_sV = self.frame_s_frame(
            stokes_component_index=3,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            angles=angles,
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
        )

        epsilonI = self.epsilon(eta_s=eta_rho_sI, nu=self.nu)
        epsilonQ = self.epsilon(eta_s=eta_rho_sQ, nu=self.nu)
        epsilonU = self.epsilon(eta_s=eta_rho_sU, nu=self.nu)
        epsilonV = self.epsilon(eta_s=eta_rho_sV, nu=self.nu)

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

    def phi(self, nui, nu, macroscopic_velocity_cm_sm1, delta_v_thermal_cm_sm1, voigt_a):
        """
        Reference: (5.43 - 5.45)
        """
        delta_nu_D = nui * delta_v_thermal_cm_sm1 / c_cm_sm1  # Doppler width

        nu_round = (nui - nu) / delta_nu_D  # nui already accounts for magnetic shifts
        nu_round_A = macroscopic_velocity_cm_sm1 / delta_v_thermal_cm_sm1

        complex_voigt = voigt(nu=nu_round - nu_round_A, a=voigt_a) / sqrt_pi / delta_nu_D
        return complex_voigt

    def cutoff_condition(self, term_upper: Term, term_lower: Term, nu: np.ndarray):
        nuimax = energy_cmm1_to_frequency_hz(term_upper.get_max_energy_cmm1() - term_lower.get_min_energy_cmm1())
        nuimin = energy_cmm1_to_frequency_hz(term_upper.get_min_energy_cmm1() - term_lower.get_max_energy_cmm1())
        cutoff = self.delta_nu_cutoff
        if min(nu) > nuimax + cutoff or max(nu) < nuimin - cutoff:
            logging.info(f"Cutoff condition: nui=[{nuimin}...{nuimax}], nu=[{min(nu)}...{max(nu)}]")
            return True
        return False

    @staticmethod
    def epsilon(eta_s: np.ndarray, nu: np.ndarray):
        """
        Reference:
        (7.47e)
        """
        return 2 * h_erg_s * nu**3 / c_cm_sm1**2 * np.real(eta_s)
