try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy import exp, pi, sqrt

from solrat.atom_model.base_atom_model.statistical_equilibrium_equations import BaseSEE
from solrat.atom_model.multi_level_atom_model.object.atmosphere_parameters import AtmosphereParameters
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.radiation_tensor import RadiationTensor
from solrat.atom_model.multi_level_atom_model.object.rho_matrix_builder import (
    Rho,
    RhoMatrixBuilder,
    construct_coherence_id_from_level_id,
)
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, kB_erg_Km1
from solrat.atom_model.shared.utility.wigner_3j_6j_9j import wigner_3j, wigner_6j, wigner_9j
from solrat.engine.functions.decorators import log_method
from solrat.engine.functions.general import m1p, n_proj
from solrat.engine.generators.merge_frame import Frame, SumLimits
from solrat.engine.generators.merge_loopers import DummyOrAlreadyMerged, FromTo, Intersection, Projection, Triangular


class MultiLevelAtomSEE(BaseSEE):
    r"""
    Statistical Equilibrium Equations within the Multi-Level atom model.

    :param level_registry:  :class:`LevelRegistry` instance for the multi-level atom under study.
    :param transition_registry:  :class:`TransitionRegistry` instance for the multi-level atom under study.
    :param disable_r_s:  DEPRECATED, scheduled for removal. Disables only the stimulated-emission
        relaxation :math:`R_S`, not the transfer :math:`T_S` nor the RTE stimulated-emission opacity;
        use the Wien limit (large :math:`h\nu_0 / k T`) to remove stimulated emission consistently.
    :param collisions:  Optional :class:`ParametrizedCollisions` (not-yet-validated feature);
        ``None`` means collisionless (pure scattering).

    Reference: (LL04 7.11, 7.14a-f); collisional rates (LL04 7.13, eq. 7.101).
    """

    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        disable_r_s: bool = False,
        collisions: Optional[ParametrizedCollisions] = None,
    ):
        self.level_registry: LevelRegistry = level_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.matrix_builder: RhoMatrixBuilder = RhoMatrixBuilder(levels=list(self.level_registry.levels.values()))
        self.disable_r_s = disable_r_s
        self.collisions = collisions

        # Precomputed frames:
        self.coherence_decay_frame: Union[Frame, None] = None
        self.absorption_frame: Union[Frame, None] = None
        self.emission_e_frame: Union[Frame, None] = None
        self.emission_s_frame: Union[Frame, None] = None
        self.relaxation_e_frame: Union[Frame, None] = None
        self.relaxation_a_frame: Union[Frame, None] = None
        self.relaxation_s_frame: Union[Frame, None] = None

    @classmethod
    def from_model_config(cls, config: MultiLevelAtomConfig) -> Self:
        logging.info("Constructing MultiLevelAtomSEE instance")
        return cls(
            level_registry=config.level_registry,
            transition_registry=config.transition_registry,
            disable_r_s=config.disable_r_s,
            collisions=config.collisions,
        )

    def _base_frame_levels(self) -> pd.DataFrame:
        """One row per level with columns level_id, J, g, expanded over K, Q."""

        class BaseSumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            K = Triangular(J, J)
            Q = Projection(K)

        rows = [
            {"level_id": level.level_id, "J": level.J, "g": level.g} for level in self.level_registry.levels.values()
        ]
        return Frame.from_sum_limits(base_frame=pd.DataFrame(rows), sum_limits=BaseSumLimits()).frame

    def _base_frame_transitions_lower(self) -> pd.DataFrame:
        """One row per transition; level_id/J/g are for the *upper* level (equation side),
        expanded over K, Q of the upper level."""

        class BaseLowerSumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            level_lower_id = DummyOrAlreadyMerged()
            Jl = DummyOrAlreadyMerged(level_lower_id)
            gl = DummyOrAlreadyMerged(level_lower_id)
            transition_id = DummyOrAlreadyMerged()
            K = Triangular(J, J)
            Q = Projection(K)

        rows = []
        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower
            rows.append(
                {
                    "level_id": level_upper.level_id,
                    "J": level_upper.J,
                    "g": level_upper.g,
                    "level_lower_id": level_lower.level_id,
                    "Jl": level_lower.J,
                    "gl": level_lower.g,
                    "transition_id": transition.transition_id,
                }
            )
        return Frame.from_sum_limits(base_frame=pd.DataFrame(rows), sum_limits=BaseLowerSumLimits()).frame

    def _base_frame_transitions_upper(self) -> pd.DataFrame:
        """One row per transition; level_id/J/g are for the *lower* level (equation side),
        expanded over K, Q of the lower level."""

        class BaseUpperSumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            level_upper_id = DummyOrAlreadyMerged()
            Ju = DummyOrAlreadyMerged(level_upper_id)
            gu = DummyOrAlreadyMerged(level_upper_id)
            transition_id = DummyOrAlreadyMerged()
            K = Triangular(J, J)
            Q = Projection(K)

        rows = []
        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower
            rows.append(
                {
                    "level_id": level_lower.level_id,
                    "J": level_lower.J,
                    "g": level_lower.g,
                    "level_upper_id": level_upper.level_id,
                    "Ju": level_upper.J,
                    "gu": level_upper.g,
                    "transition_id": transition.transition_id,
                }
            )
        return Frame.from_sum_limits(base_frame=pd.DataFrame(rows), sum_limits=BaseUpperSumLimits()).frame

    @log_method
    def fill_all_equations(
        self,
        atmosphere_parameters: AtmosphereParameters,
        radiation_tensor_in_magnetic_frame: RadiationTensor,
    ):
        r"""
        Loop through all equations to construct the complete system of equations for rho.

        Reference: (LL04 7.11)
        """
        self.matrix_builder.reset_matrix()
        self.add_coherence_decay(atmosphere_parameters=atmosphere_parameters)
        self.add_absorption(radiation_tensor=radiation_tensor_in_magnetic_frame)
        self.add_emission_e()
        self.add_emission_s(radiation_tensor=radiation_tensor_in_magnetic_frame)
        self.add_relaxation_e()
        self.add_relaxation_a(radiation_tensor=radiation_tensor_in_magnetic_frame)
        self.add_relaxation_s(radiation_tensor=radiation_tensor_in_magnetic_frame)
        if self.collisions is not None:
            self.add_collisions(atmosphere_parameters=atmosphere_parameters)

    @log_method
    def add_coherence_decay(self, atmosphere_parameters: AtmosphereParameters):
        r"""
        Add the Larmor coherence decay :math:`-2\pi i \nu_L g_{\alpha J} Q \rho^K_Q(\alpha J)`.

        Reference: (LL04 7.11, first term)
        """

        class CoherenceDecaySumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()

        if self.coherence_decay_frame is None:
            self.coherence_decay_frame = (
                Frame.from_sum_limits(
                    self._base_frame_levels(),
                    CoherenceDecaySumLimits(),
                )
                .register_multiplication(lambda g, Q: g * Q, elementwise=True)
                .to_coefficient()
            )

        self.add_coefficient_for_rho(
            frame=self.coherence_decay_frame,
            multiply_by=-2 * pi * 1j * atmosphere_parameters.nu_larmor,
            level_id="level_id",
            K="K",
            Q="Q",
        )

    @log_method
    def add_absorption(self, radiation_tensor: RadiationTensor):
        r"""
        Add absorption :math:`T_A`.

        Reference: (LL04 7.11, 7.14a)
        """

        class AbsorptionSumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            level_lower_id = DummyOrAlreadyMerged()
            Jl = DummyOrAlreadyMerged(level_lower_id)
            gl = DummyOrAlreadyMerged(level_lower_id)
            transition_id = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Kl = Triangular(Jl, Jl)
            Ql = Projection(Kl)
            Kr = Intersection(FromTo(0, 2), Triangular(K, Kl))
            Qr = Intersection(Projection(Kr), Ql - Q)

        if self.absorption_frame is None:
            self.absorption_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_lower(),
                    AbsorptionSumLimits(),
                )
                .register_multiplication(
                    lambda transition_id, J, Jl, K, Kl, Kr, Q, Ql, Qr: (
                        (2 * Jl + 1)
                        * self.transition_registry.transitions[str(transition_id)].einstein_b_lu
                        * sqrt(3 * n_proj(K, Kl, Kr))
                        * m1p(Kl + Ql)
                        * wigner_9j(J, Jl, 1, J, Jl, 1, K, Kl, Kr)
                        * wigner_3j(K, Kl, Kr, -Q, Ql, -Qr)
                    ),
                    elementwise=True,
                )
                .to_coefficient()
            )

        absorption_frame = self.absorption_frame.copy()
        absorption_frame.register_multiplication(
            lambda transition_id, Kr, Qr: radiation_tensor.get_from_transition_id(
                transition_id=transition_id, K=Kr, Q=Qr
            ),
            elementwise=True,
        )
        absorption_frame.reduce_partially("transition_id", "Kr", "Qr")

        self.add_coefficient_for_rho(
            frame=absorption_frame,
            level_id="level_lower_id",
            K="Kl",
            Q="Ql",
        )

    @log_method
    def add_emission_e(self):
        r"""
        Add spontaneous emission :math:`T_E`.

        Diagonal in :math:`K, Q` (:math:`K_u = K, Q_u = Q`).

        Reference: (LL04 7.11, 7.14b)
        """

        class EmissionESumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            level_upper_id = DummyOrAlreadyMerged()
            Ju = DummyOrAlreadyMerged(level_upper_id)
            gu = DummyOrAlreadyMerged(level_upper_id)
            transition_id = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Ku = Intersection(Triangular(Ju, Ju), K)
            Qu = Intersection(Projection(Ku), Q)

        if self.emission_e_frame is None:
            self.emission_e_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_upper(),
                    EmissionESumLimits(),
                )
                .register_multiplication(
                    lambda transition_id, J, Ju, K: (
                        (2 * Ju + 1)
                        * self.transition_registry.transitions[str(transition_id)].einstein_a_ul
                        * m1p(1 + J + Ju + K)
                        * wigner_6j(Ju, Ju, K, J, J, 1)
                    ),
                    elementwise=True,
                )
                .reduce_partially("transition_id")
            )

        self.add_coefficient_for_rho(
            frame=self.emission_e_frame,
            level_id="level_upper_id",
            K="Ku",
            Q="Qu",
        )

    @log_method
    def add_emission_s(self, radiation_tensor: RadiationTensor):
        r"""
        Add stimulated emission :math:`T_S`.

        Reference: (LL04 7.11, 7.14c)
        """

        class EmissionSSumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            level_upper_id = DummyOrAlreadyMerged()
            Ju = DummyOrAlreadyMerged(level_upper_id)
            gu = DummyOrAlreadyMerged(level_upper_id)
            transition_id = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Ku = Triangular(Ju, Ju)
            Qu = Projection(Ku)
            Kr = Intersection(FromTo(0, 2), Triangular(K, Ku))
            Qr = Intersection(Projection(Kr), Qu - Q)

        if self.emission_s_frame is None:
            self.emission_s_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_upper(),
                    EmissionSSumLimits(),
                )
                .register_multiplication(
                    lambda transition_id, J, Ju, K, Ku, Kr, Q, Qu, Qr: (
                        (2 * Ju + 1)
                        * self.transition_registry.transitions[str(transition_id)].einstein_b_ul
                        * sqrt(3 * n_proj(K, Ku, Kr))
                        * m1p(Kr + Ku + Qu)
                        * wigner_9j(J, Ju, 1, J, Ju, 1, K, Ku, Kr)
                        * wigner_3j(K, Ku, Kr, -Q, Qu, -Qr)
                    ),
                    elementwise=True,
                )
                .to_coefficient()
            )

        emission_s_frame = self.emission_s_frame.copy()
        emission_s_frame.register_multiplication(
            lambda transition_id, Kr, Qr: radiation_tensor.get_from_transition_id(
                transition_id=transition_id, K=Kr, Q=Qr
            ),
            elementwise=True,
        )
        emission_s_frame.reduce_partially("transition_id", "Kr", "Qr")

        self.add_coefficient_for_rho(
            frame=emission_s_frame,
            level_id="level_upper_id",
            K="Ku",
            Q="Qu",
        )

    @log_method
    def add_relaxation_e(self):
        r"""
        Add spontaneous emission relaxation :math:`R_E = -\sum_{\alpha_l J_l} A(\alpha J \to \alpha_l J_l)`.

        Diagonal in :math:`K, Q` (here we explicitly resolve summation over :math:`Kʹ, Qʹ`.

        Reference: (LL04 7.11, 7.14e)
        """

        class RelaxationESumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            level_lower_id = DummyOrAlreadyMerged()
            Jl = DummyOrAlreadyMerged(level_lower_id)
            gl = DummyOrAlreadyMerged(level_lower_id)
            transition_id = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()

        if self.relaxation_e_frame is None:
            self.relaxation_e_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_lower(),
                    RelaxationESumLimits(),
                )
                .register_multiplication(
                    lambda transition_id: self.transition_registry.transitions[str(transition_id)].einstein_a_ul,
                    elementwise=True,
                )
                .reduce_partially("level_lower_id", "Jl", "gl", "transition_id")
            )

        self.add_coefficient_for_rho(
            frame=self.relaxation_e_frame,
            multiply_by=-1,
            level_id="level_id",
            K="K",
            Q="Q",
        )

    @log_method
    def add_relaxation_a(self, radiation_tensor: RadiationTensor):
        r"""
        Add absorption relaxation :math:`R_A`.

        Reference: (LL04 7.11, 7.14d)
        """

        class RelaxationASumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            level_upper_id = DummyOrAlreadyMerged()
            Ju = DummyOrAlreadyMerged(level_upper_id)
            gu = DummyOrAlreadyMerged(level_upper_id)
            transition_id = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Kʹ = Triangular(J, J)
            Qʹ = Projection(Kʹ)
            Kr = Intersection(FromTo(0, 2), Triangular(K, Kʹ))
            Qr = Intersection(Projection(Kr), Qʹ - Q)

        if self.relaxation_a_frame is None:
            self.relaxation_a_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_upper(),
                    RelaxationASumLimits(),
                )
                .register_multiplication(
                    lambda transition_id, J, Ju, K, Kʹ, Kr, Q, Qʹ, Qr: (
                        (2 * J + 1)
                        * self.transition_registry.transitions[str(transition_id)].einstein_b_lu
                        * sqrt(3 * n_proj(K, Kʹ, Kr))
                        * m1p(1 + Ju - J + Kr + Qʹ)
                        * wigner_6j(K, Kʹ, Kr, J, J, J)
                        * wigner_6j(1, 1, Kr, J, J, Ju)
                        * wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr)
                        * 0.5
                        * (1 + m1p(K + Kʹ + Kr))
                    ),
                    elementwise=True,
                )
                .to_coefficient()
            )

        relaxation_a_frame = self.relaxation_a_frame.copy()
        relaxation_a_frame.register_multiplication(
            lambda transition_id, Kr, Qr: radiation_tensor.get_from_transition_id(
                transition_id=transition_id, K=Kr, Q=Qr
            ),
            elementwise=True,
        )
        relaxation_a_frame.reduce_partially("level_upper_id", "Ju", "gu", "transition_id", "Kr", "Qr")

        self.add_coefficient_for_rho(
            frame=relaxation_a_frame,
            multiply_by=-1,
            level_id="level_id",
            K="Kʹ",
            Q="Qʹ",
        )

    @log_method
    def add_relaxation_s(self, radiation_tensor: RadiationTensor):
        r"""
        Add stimulated emission relaxation :math:`R_S`.

        Reference: (LL04 7.11, 7.14f)
        """
        # DEPRECATED: disable_r_s gates only this R_S term (not T_S or the RTE eta_S), so it is not a
        # consistent stimulated-emission switch. Scheduled for removal (warned at config construction).
        if self.disable_r_s:
            return

        class RelaxationSSumLimits(SumLimits):
            level_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged(level_id)
            g = DummyOrAlreadyMerged(level_id)
            level_lower_id = DummyOrAlreadyMerged()
            Jl = DummyOrAlreadyMerged(level_lower_id)
            gl = DummyOrAlreadyMerged(level_lower_id)
            transition_id = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Kʹ = Triangular(J, J)
            Qʹ = Projection(Kʹ)
            Kr = Intersection(FromTo(0, 2), Triangular(K, Kʹ))
            Qr = Intersection(Projection(Kr), Qʹ - Q)

        if self.relaxation_s_frame is None:
            self.relaxation_s_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_lower(),
                    RelaxationSSumLimits(),
                )
                .register_multiplication(
                    lambda transition_id, J, Jl, K, Kʹ, Kr, Q, Qʹ, Qr: (
                        (2 * J + 1)
                        * self.transition_registry.transitions[str(transition_id)].einstein_b_ul
                        * sqrt(3 * n_proj(K, Kʹ, Kr))
                        * m1p(1 + Jl - J + Qʹ)
                        * wigner_6j(K, Kʹ, Kr, J, J, J)
                        * wigner_6j(1, 1, Kr, J, J, Jl)
                        * wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr)
                        * 0.5
                        * (1 + m1p(K + Kʹ + Kr))
                    ),
                    elementwise=True,
                )
                .to_coefficient()
            )

        relaxation_s_frame = self.relaxation_s_frame.copy()
        relaxation_s_frame.register_multiplication(
            lambda transition_id, Kr, Qr: radiation_tensor.get_from_transition_id(
                transition_id=transition_id, K=Kr, Q=Qr
            ),
            elementwise=True,
        )
        relaxation_s_frame.reduce_partially("level_lower_id", "Jl", "gl", "transition_id", "Kr", "Qr")

        self.add_coefficient_for_rho(
            frame=relaxation_s_frame,
            multiply_by=-1,
            level_id="level_id",
            K="Kʹ",
            Q="Qʹ",
        )

    @log_method
    def add_collisions(self, atmosphere_parameters: AtmosphereParameters):
        r"""
        Add parametrized inelastic/superelastic and elastic (depolarizing) collisional rates.

        Implements LL04 eq. (7.101) in the spherical-statistical-tensor representation: collisions
        couple only equal (K, Q). Per radiative transition (upper, lower) the user supplies the
        superelastic de-excitation rate :math:`C_{ul}` (:math:`C_S^{(0)}(l, u)`); the inelastic
        excitation rate :math:`C_{lu}` (:math:`C_I^{(0)}(u, l)`) follows from Einstein-Milne
        detailed balance (LL04 eq. 7.98):
        :math:`C_{lu} = \frac{2J_u + 1}{2J_l + 1}\, e^{-(E_u - E_l)/(k_B T)}\, C_{ul}`.
        Transfer multipole components are taken K-independent, :math:`C^{(K)} = C^{(0)}`. Elastic
        collisions add the depolarizing loss :math:`-D^{(K)}` for K >= 1 (LL04 eq. 7.102).
        Collisional rates simply add to the radiative rates (LL04 Sec. 7.13.e).

        Reference: (LL04 7.101, 7.98, 7.102).
        """
        assert self.collisions is not None, "add_collisions called without configured collisions."
        temperature_K = atmosphere_parameters.temperature_K

        rows = []
        for transition in self.transition_registry.transitions.values():
            upper = transition.level_upper
            lower = transition.level_lower
            c_ul = self.collisions.deexcitation_rate_sm1(transition.transition_id)  # C_S^(0)(l, u)
            if c_ul <= 0:
                continue
            delta_e_erg = (upper.energy_cmm1 - lower.energy_cmm1) * h_erg_s * c_cm_sm1
            c_lu = (2 * upper.J + 1) / (2 * lower.J + 1) * exp(-delta_e_erg / (kB_erg_Km1 * temperature_K)) * c_ul

            # Transfer terms (diagonal in K, Q; K valid for both levels), LL04 (7.101).
            factor_into_upper = sqrt((2 * lower.J + 1) / (2 * upper.J + 1)) * c_lu
            factor_into_lower = sqrt((2 * upper.J + 1) / (2 * lower.J + 1)) * c_ul
            for K in range(0, int(2 * min(upper.J, lower.J)) + 1):
                for Q in range(-K, K + 1):
                    rows.append(self._collision_row(upper.level_id, K, Q, lower.level_id, K, Q, factor_into_upper))
                    rows.append(self._collision_row(lower.level_id, K, Q, upper.level_id, K, Q, factor_into_lower))

            # Relaxation (loss) of each level to its transition partner, LL04 (7.101).
            for K in range(0, int(2 * upper.J) + 1):
                for Q in range(-K, K + 1):
                    rows.append(self._collision_row(upper.level_id, K, Q, upper.level_id, K, Q, -c_ul))
            for K in range(0, int(2 * lower.J) + 1):
                for Q in range(-K, K + 1):
                    rows.append(self._collision_row(lower.level_id, K, Q, lower.level_id, K, Q, -c_lu))

        # Elastic depolarizing loss D^(K), K >= 1, per level, LL04 (7.102).
        for level in self.level_registry.levels.values():
            for K in range(1, int(2 * level.J) + 1):
                d_k = self.collisions.depolarizing_rate_sm1(level.level_id, K)
                if d_k <= 0:
                    continue
                for Q in range(-K, K + 1):
                    rows.append(self._collision_row(level.level_id, K, Q, level.level_id, K, Q, -d_k))

        if not rows:
            return
        df = pd.DataFrame(rows)
        df = self.add_equation_index(df, level_id="eq_level_id", K="eq_K", Q="eq_Q", index="index0")
        df = self.add_equation_index(df, level_id="rho_level_id", K="rho_K", Q="rho_Q", index="index1")
        self.matrix_builder.add_coefficient_from_df(df)

    @staticmethod
    def _collision_row(eq_level_id, eq_K, eq_Q, rho_level_id, rho_K, rho_Q, coefficient):
        r"""
        One collisional matrix entry: the coefficient multiplying rho^K_Q(rho_level) in the
        equation for rho^K_Q(eq_level).
        """
        return {
            "eq_level_id": eq_level_id,
            "eq_K": eq_K,
            "eq_Q": eq_Q,
            "rho_level_id": rho_level_id,
            "rho_K": rho_K,
            "rho_Q": rho_Q,
            "coefficient": coefficient,
        }

    @log_method
    def get_solution(self) -> Rho:
        r"""
        Get the solution of the Statistical Equilibrium Equations.

        :return: :class:`Rho` instance.

        See multi-term :meth:`get_solution` for details on the manual linalg approach.
        """
        # logging.info("Solving Statistical Equilibrium Equations")
        sol = -np.linalg.pinv(self.matrix_builder.rho_matrix[1:, 1:]) @ self.matrix_builder.rho_matrix[1:, 0:1]
        sol = np.insert(sol, 0, 1.0, 0)
        sol = sol[:, 0]

        # Normalize: Sum sqrt(2J+1) rho00(alpha J) = 1
        weights = np.zeros_like(sol)
        for index, weight in zip(self.matrix_builder.trace_indexes, self.matrix_builder.trace_weights):
            weights[index] = weight
        trace = (sol * weights).sum()

        rho_vector = sol / trace

        rho = Rho(levels=list(self.level_registry.levels.values()))
        for index, (level_id, k, q) in self.matrix_builder.index_to_parameters.items():
            rho.set_from_level_id(level_id=level_id, K=k, Q=q, value=rho_vector[index])

        return rho

    def add_coefficient_for_rho(
        self,
        frame: Frame,
        level_id: str,
        K: str,
        Q: str,
        multiply_by: Union[complex, float, None] = None,
    ):
        df = frame.frame.copy()
        df = self.add_equation_index(df, level_id="level_id", K="K", Q="Q", index="index0")
        df = self.add_equation_index(df, level_id=level_id, K=K, Q=Q, index="index1")
        if multiply_by is not None:
            df["coefficient"] = df["coefficient"] * multiply_by
        self.matrix_builder.add_coefficient_from_df(df)

    def add_equation_index(self, df: pd.DataFrame, level_id: str, K: str, Q: str, index: str):
        """
        A helper function to keep track of which matrix row/column each term in SEE corresponds to.
        Sets either index0 or index1 using the provided level_id, K, Q column names.
        """
        df[index] = df.apply(
            lambda row: self.matrix_builder.coherence_id_to_index[
                construct_coherence_id_from_level_id(level_id=row[level_id], K=row[K], Q=row[Q])
            ],
            axis=1,
        )
        return df
