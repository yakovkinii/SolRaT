try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import logging
from typing import Union

import numpy as np
import pandas as pd
from numpy import pi, sqrt

from solrat.atom_model.base_atom_model.statistical_equilibrium_equations import BaseSEE
from solrat.atom_model.multi_term_atom_model.object.atmosphere_parameters import AtmosphereParameters
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.radiation_tensor import RadiationTensor
from solrat.atom_model.multi_term_atom_model.object.rho_matrix_builder import Rho, RhoMatrixBuilder
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.utility.functions import energy_cmm1_to_frequency_sm1
from solrat.atom_model.shared.utility.wigner_3j_6j_9j import wigner_3j, wigner_6j, wigner_9j
from solrat.engine.functions.decorators import log_method
from solrat.engine.functions.general import delta, m1p, n_proj
from solrat.engine.generators.merge_frame import Frame, SumLimits
from solrat.engine.generators.merge_loopers import DummyOrAlreadyMerged, FromTo, Intersection, Projection, Triangular


class MultiTermAtomSEE(BaseSEE):
    r"""
    Statistical Equilibrium Equations within Multi-Term atom model

    :param level_registry:  LevelRegistry instance for the multi-term atom under study.
    :param transition_registry:  TransitionRegistry instance for the multi-term atom under study.
    :param disable_r_s:  DEPRECATED, scheduled for removal. Disables only the stimulated-emission
        relaxation :math:`R_S` (7.46c), not the transfer :math:`T_S` nor the RTE stimulated-emission
        opacity; use the Wien limit (large :math:`h\nu_0 / k T`) to remove it consistently.

    Reference: (LL04 7.38)

    Note on precomputing:
    All frames are built lazily on first use and cached as instance attributes.
    Atom-specific frames (no radiation tensor) are built once and reused across calls.
    Frames that depend on the radiation tensor are partially precomputed; the radiation-tensor
    factor is applied per :meth:`fill_all_equations` call.
    Precomputed frames can be extracted after the first :meth:`fill_all_equations` call and
    injected into a new instance via :class:`PrecomputedData` to skip the build step.
    """

    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        disable_r_s: bool = False,
    ):
        self.level_registry: LevelRegistry = level_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.matrix_builder: RhoMatrixBuilder = RhoMatrixBuilder(terms=list(self.level_registry.terms.values()))

        self.disable_r_s = disable_r_s

        # Precomputed frames:
        self.coherence_decay_frame_n_0: Union[Frame, None] = None
        self.coherence_decay_frame_n_1: Union[Frame, None] = None
        self.absorption_frame: Union[Frame, None] = None
        self.emission_e_frame: Union[Frame, None] = None
        self.emission_s_frame: Union[Frame, None] = None
        self.relaxation_e_frame: Union[Frame, None] = None
        self.relaxation_a_frame: Union[Frame, None] = None
        self.relaxation_s_frame: Union[Frame, None] = None

    @classmethod
    def from_model_config(cls, config: MultiTermAtomConfig) -> Self:
        r"""
        Constructor from the model config.
        """
        logging.info("Constructing MultiTermAtomSEE instance")

        if config.precomputed_data is None:
            return cls(
                level_registry=config.level_registry,
                transition_registry=config.transition_registry,
                disable_r_s=config.disable_r_s,
            )

        see = cls(
            level_registry=config.level_registry,
            transition_registry=config.transition_registry,
            disable_r_s=config.disable_r_s,
        )
        see.coherence_decay_frame_n_0 = config.precomputed_data.coherence_decay_frame
        see.coherence_decay_frame_n_1 = config.precomputed_data.coherence_decay_frame_n_1
        see.absorption_frame = config.precomputed_data.absorption_frame
        see.emission_e_frame = config.precomputed_data.emission_e_frame
        see.emission_s_frame = config.precomputed_data.emission_s_frame
        see.relaxation_e_frame = config.precomputed_data.relaxation_e_frame
        see.relaxation_a_frame = config.precomputed_data.relaxation_a_frame
        see.relaxation_s_frame = config.precomputed_data.relaxation_s_frame
        return see

    def _base_frame_terms(self) -> pd.DataFrame:
        """One row per term with columns term_id, L, S, expanded over J, Jʹ, K, Q."""

        class BaseSumLimits(SumLimits):
            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged(term_id)
            S = DummyOrAlreadyMerged(term_id)
            J = Triangular(L, S)
            Jʹ = Triangular(L, S)
            K = Triangular(J, Jʹ)
            Q = Projection(K)

        frame = Frame.from_sum_limits(
            base_frame=pd.DataFrame(
                [{"term_id": term.term_id, "L": term.L, "S": term.S} for term in self.level_registry.terms.values()]
            ),
            sum_limits=BaseSumLimits(),
        )

        return frame.frame

    def _base_frame_transitions_lower(self) -> pd.DataFrame:
        """One row per transition; term_id/L/S are for the *upper* term (equation side),
        expanded over J, Jʹ, K, Q of the upper term."""

        class BaseLowerSumLimits(SumLimits):
            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged(term_id)
            S = DummyOrAlreadyMerged(term_id)
            term_lower_id = DummyOrAlreadyMerged()
            Ll = DummyOrAlreadyMerged(term_lower_id)
            transition_id = DummyOrAlreadyMerged()
            J = Triangular(L, S)
            Jʹ = Triangular(L, S)
            K = Triangular(J, Jʹ)
            Q = Projection(K)

        rows = []
        for t_upper in self.level_registry.terms.values():
            for t_lower in self.level_registry.terms.values():
                if not self.transition_registry.is_transition_registered(term_upper=t_upper, term_lower=t_lower):
                    continue
                tr = self.transition_registry.get_transition(term_upper=t_upper, term_lower=t_lower)
                rows.append(
                    {
                        "term_id": t_upper.term_id,
                        "L": t_upper.L,
                        "S": t_upper.S,
                        "term_lower_id": t_lower.term_id,
                        "Ll": t_lower.L,
                        "transition_id": tr.transition_id,
                    }
                )
        frame = Frame.from_sum_limits(
            base_frame=pd.DataFrame(rows),
            sum_limits=BaseLowerSumLimits(),
        )

        return frame.frame

    def _base_frame_transitions_upper(self) -> pd.DataFrame:
        """One row per transition; term_id/L/S are for the *lower* term (equation side),
        expanded over J, Jʹ, K, Q of the lower term."""

        class BaseUpperSumLimits(SumLimits):
            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged(term_id)
            S = DummyOrAlreadyMerged(term_id)
            term_upper_id = DummyOrAlreadyMerged()
            Lu = DummyOrAlreadyMerged(term_upper_id)
            transition_id = DummyOrAlreadyMerged()
            J = Triangular(L, S)
            Jʹ = Triangular(L, S)
            K = Triangular(J, Jʹ)
            Q = Projection(K)

        rows = []
        for t_upper in self.level_registry.terms.values():
            for t_lower in self.level_registry.terms.values():
                if not self.transition_registry.is_transition_registered(term_upper=t_upper, term_lower=t_lower):
                    continue
                tr = self.transition_registry.get_transition(term_upper=t_upper, term_lower=t_lower)
                rows.append(
                    {
                        "term_id": t_lower.term_id,
                        "L": t_lower.L,
                        "S": t_lower.S,
                        "term_upper_id": t_upper.term_id,
                        "Lu": t_upper.L,
                        "transition_id": tr.transition_id,
                    }
                )
        frame = Frame.from_sum_limits(
            base_frame=pd.DataFrame(rows),
            sum_limits=BaseUpperSumLimits(),
        )

        return frame.frame

    @log_method
    def fill_all_equations(  # Main entry point
        self,
        atmosphere_parameters: AtmosphereParameters,
        radiation_tensor_in_magnetic_frame: RadiationTensor,
    ):
        r"""
        Loop through all equations to construct the complete system of equations for rho.

        :param atmosphere_parameters:  AtmosphereParameters instance carrying the magnetic field and other variables.
        :param radiation_tensor_in_magnetic_frame:  RadiationTensor instance

        Reference: (LL04 7.38)

        Note:
        Calling this function multiple times with different atmosphere_parameters of :math:`J` tensors is safe,
        it will build the system of equations from scratch each time.
        """
        self.matrix_builder.reset_matrix()
        self.add_coherence_decay(atmosphere_parameters=atmosphere_parameters)
        self.add_absorption(radiation_tensor=radiation_tensor_in_magnetic_frame)
        self.add_emission_e()
        self.add_emission_s(radiation_tensor=radiation_tensor_in_magnetic_frame)
        self.add_relaxation_e()
        self.add_relaxation_a(radiation_tensor=radiation_tensor_in_magnetic_frame)
        self.add_relaxation_s(radiation_tensor=radiation_tensor_in_magnetic_frame)

    @log_method
    def add_coherence_decay(self, atmosphere_parameters: AtmosphereParameters):
        r"""
        Add the coherence decay :math:`N = n_0 + n_1 \nu_L`

        Reference: (LL04 7.38, 7.41)
        """

        class CoherenceDecaySumLimits(SumLimits):
            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged()
            S = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged()
            Jʹ = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()

            Jʹʹ = Triangular(L, S)
            Jʹʹʹ = Triangular(L, S)
            Kʹ = Intersection(Triangular(K, 1), Triangular(Jʹʹ, Jʹʹʹ))
            Qʹ = Intersection(Projection(Kʹ), Q)

        if self.coherence_decay_frame_n_0 is None:
            self.coherence_decay_frame_n_0 = (
                Frame.from_sum_limits(
                    self._base_frame_terms(),
                    CoherenceDecaySumLimits(),
                )
                .register_multiplication(
                    lambda K, Kʹ, J, Jʹ, Jʹʹ, Jʹʹʹ: delta(K, Kʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ),
                    lambda term_id, J, Jʹ: energy_cmm1_to_frequency_sm1(
                        self.level_registry.terms[term_id].get_level(J).energy_cmm1
                        - self.level_registry.terms[term_id].get_level(Jʹ).energy_cmm1
                    ),
                    elementwise=True,
                )
                .reduce_partially("Jʹʹ", "Jʹʹʹ", "Kʹ", "Qʹ")
            )

        self.add_coefficient_for_rho(
            frame=self.coherence_decay_frame_n_0,
            multiply_by=-2 * pi * 1j,
            term_id="term_id",
            K="K",
            Q="Q",
            J="J",
            Jʹ="Jʹ",
        )

        if self.coherence_decay_frame_n_1 is None:

            def _gamma(L, S, J, Jʹ):
                return delta(J, Jʹ) * sqrt(J * (J + 1) * (2 * J + 1)) + (
                    m1p(1 + L + S + J)
                    * sqrt((2 * J + 1) * (2 * Jʹ + 1) * S * (S + 1) * (2 * S + 1))
                    * wigner_6j(J, Jʹ, 1, S, S, L)
                )

            self.coherence_decay_frame_n_1 = (
                Frame.from_sum_limits(
                    self._base_frame_terms(),
                    CoherenceDecaySumLimits(),
                )
                .register_multiplication(
                    lambda K, Kʹ, Q, J, Jʹ, Jʹʹ, Jʹʹʹ, L, S: (
                        m1p(J + Jʹ - Q)
                        * sqrt((2 * K + 1) * (2 * Kʹ + 1))
                        * wigner_3j(K, Kʹ, 1, -Q, Q, 0)
                        * (
                            delta(Jʹ, Jʹʹʹ) * _gamma(L, S, J, Jʹʹ) * wigner_6j(K, Kʹ, 1, Jʹʹ, J, Jʹ)
                            + delta(J, Jʹʹ) * m1p(K - Kʹ) * _gamma(L, S, Jʹʹʹ, Jʹ) * wigner_6j(K, Kʹ, 1, Jʹʹʹ, Jʹ, J)
                        )
                    ),
                    elementwise=True,
                )
                .to_coefficient()
            )

        self.add_coefficient_for_rho(
            frame=self.coherence_decay_frame_n_1,
            multiply_by=-2 * pi * 1j * atmosphere_parameters.nu_larmor,
            term_id="term_id",
            K="Kʹ",
            Q="Qʹ",
            J="Jʹʹ",
            Jʹ="Jʹʹʹ",
        )

    @log_method
    def add_absorption(self, radiation_tensor: RadiationTensor):
        r"""
        Add absorption :math:`T_a = t_{a1} \cdot J^{K_r}_{Q_r}`.

        Reference: (LL04 7.38, 7.45a)
        """

        class AbsorptionSumLimits(SumLimits):
            r"""
            Summation limits for absorption :math:`T_a = t_{a1} \cdot J^{K_r}_{Q_r}`.
            """

            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged(term_id)
            S = DummyOrAlreadyMerged(term_id)
            term_lower_id = DummyOrAlreadyMerged()
            Ll = DummyOrAlreadyMerged()
            transition_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged()
            Jʹ = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Jl = Intersection(Triangular(Ll, S), Triangular(J, 1))
            Jʹl = Intersection(Triangular(Ll, S), Triangular(Jʹ, 1))
            Kl = Triangular(Jl, Jʹl)
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
                    lambda transition_id, J, Jʹ, Jl, Jʹl, K, Kl, Kr, Q, Ql, Qr, L, Ll, S: (
                        n_proj(Ll)
                        * self.transition_registry.transitions[str(transition_id)].einstein_b_lu
                        * sqrt(n_proj(1, J, Jʹ, Jl, Jʹl, K, Kl, Kr))
                        * m1p(Kl + Ql + Jʹl - Jl)
                        * wigner_9j(J, Jl, 1, Jʹ, Jʹl, 1, K, Kl, Kr)
                        * wigner_6j(L, Ll, 1, Jl, J, S)
                        * wigner_6j(L, Ll, 1, Jʹl, Jʹ, S)
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
            term_id="term_lower_id",
            K="Kl",
            Q="Ql",
            J="Jl",
            Jʹ="Jʹl",
        )

    @log_method
    def add_emission_e(self):
        r"""
        Add spontaneous emission :math:`T_e`.

        Reference: (LL04 7.38, 7.45b)
        """

        class EmissionESumLimits(SumLimits):
            r"""
            Summation limits for spontaneous emission :math:`T_e`.
            Diagonal in K, Q (Ku = K, Qu = Q).
            """

            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged(term_id)
            S = DummyOrAlreadyMerged(term_id)
            term_upper_id = DummyOrAlreadyMerged()
            Lu = DummyOrAlreadyMerged(term_upper_id)
            transition_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged()
            Jʹ = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Ju = Intersection(Triangular(Lu, S), Triangular(J, 1))
            Jʹu = Intersection(Triangular(Lu, S), Triangular(Jʹ, 1))
            Ku = Intersection(Triangular(Ju, Jʹu), K)
            Qu = Intersection(Projection(Ku), Q)

        if self.emission_e_frame is None:
            self.emission_e_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_upper(),
                    EmissionESumLimits(),
                )
                .register_multiplication(
                    lambda transition_id, J, Jʹ, Ju, Jʹu, K, Lu, L, S: (
                        (2 * Lu + 1)
                        * self.transition_registry.transitions[str(transition_id)].einstein_a_ul
                        * sqrt(n_proj(J, Jʹ, Ju, Jʹu))
                        * m1p(1 + K + Jʹ + Jʹu)
                        * wigner_6j(J, Jʹ, K, Jʹu, Ju, 1)
                        * wigner_6j(Lu, L, 1, J, Ju, S)
                        * wigner_6j(Lu, L, 1, Jʹ, Jʹu, S)
                    ),
                    elementwise=True,
                )
                .reduce_partially("transition_id")
            )

        self.add_coefficient_for_rho(
            frame=self.emission_e_frame,
            term_id="term_upper_id",
            K="Ku",
            Q="Qu",
            J="Ju",
            Jʹ="Jʹu",
        )

    @log_method
    def add_emission_s(self, radiation_tensor: RadiationTensor):
        r"""
        Add stimulated emission :math:`T_s = t_{s1} \cdot J^{K_r}_{Q_r}`.

        Reference: (LL04 7.38, 7.45c)
        """

        class EmissionSSumLimits(SumLimits):
            r"""
            Summation limits for stimulated emission :math:`T_s = t_{s1} \cdot J^{K_r}_{Q_r}`.
            """

            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged(term_id)
            S = DummyOrAlreadyMerged(term_id)
            term_upper_id = DummyOrAlreadyMerged()
            Lu = DummyOrAlreadyMerged(term_upper_id)
            transition_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged()
            Jʹ = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Ju = Intersection(Triangular(Lu, S), Triangular(J, 1))
            Jʹu = Intersection(Triangular(Lu, S), Triangular(Jʹ, 1))
            Ku = Triangular(Ju, Jʹu)
            Qu = Projection(Ku)
            Kr = Intersection(FromTo(0, 2), Triangular(Ku, K))
            Qr = Intersection(Projection(Kr), Qu - Q)

        if self.emission_s_frame is None:
            self.emission_s_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_upper(),
                    EmissionSSumLimits(),
                )
                .register_multiplication(
                    lambda transition_id, J, Jʹ, Ju, Jʹu, K, Ku, Kr, Q, Qu, Qr, Lu, L, S: (
                        n_proj(Lu)
                        * self.transition_registry.transitions[str(transition_id)].einstein_b_ul
                        * sqrt(n_proj(1, J, Jʹ, Ju, Jʹu, K, Ku, Kr))
                        * m1p(Kr + Ku + Qu + Jʹu - Ju)
                        * wigner_9j(J, Ju, 1, Jʹ, Jʹu, 1, K, Ku, Kr)
                        * wigner_6j(Lu, L, 1, J, Ju, S)
                        * wigner_6j(Lu, L, 1, Jʹ, Jʹu, S)
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
            term_id="term_upper_id",
            K="Ku",
            Q="Qu",
            J="Ju",
            Jʹ="Jʹu",
        )

    @log_method
    def add_relaxation_e(self):
        r"""
        Add spontaneous emission relaxation :math:`R_E = -r_{e0}`.

        Reference: (LL04 7.38, 7.46b)
        """

        class RelaxationESumLimits(SumLimits):
            r"""
            Summation limits for spontaneous emission relaxation :math:`R_E = -r_{e0}`.
            Diagonal: Kʹ = K, Qʹ = Q, Jʹʹ = J, Jʹʹʹ = Jʹ.
            """

            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged(term_id)
            S = DummyOrAlreadyMerged(term_id)
            term_lower_id = DummyOrAlreadyMerged()
            transition_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged()
            Jʹ = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Jʹʹ = Intersection(Triangular(L, S), J)
            Jʹʹʹ = Intersection(Triangular(L, S), Jʹ)
            Kʹ = Intersection(Triangular(J, Jʹ), K)
            Qʹ = Intersection(Projection(Kʹ), Q)

        if self.relaxation_e_frame is None:
            self.relaxation_e_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_lower(),
                    RelaxationESumLimits(),
                )
                .register_multiplication(
                    lambda transition_id: (self.transition_registry.transitions[str(transition_id)].einstein_a_ul),
                    elementwise=True,
                )
                .reduce_partially("Ll", "term_lower_id", "transition_id", "Jʹʹ", "Jʹʹʹ", "Kʹ", "Qʹ")
            )

        self.add_coefficient_for_rho(
            frame=self.relaxation_e_frame,
            multiply_by=-1,
            term_id="term_id",
            K="K",
            Q="Q",
            J="J",
            Jʹ="Jʹ",
        )

    @log_method
    def add_relaxation_a(self, radiation_tensor: RadiationTensor):
        r"""
        Add absorption relaxation :math:`R_A = -r_{a1} \cdot J^{K_r}_{Q_r}`.

        Reference: (LL04 7.38, 7.46a)
        """

        class RelaxationASumLimits(SumLimits):
            r"""
            Summation limits for absorption relaxation :math:`R_A = -r_{a1} \cdot J^{K_r}_{Q_r}`.
            """

            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged(term_id)
            S = DummyOrAlreadyMerged(term_id)
            term_upper_id = DummyOrAlreadyMerged()
            Lu = DummyOrAlreadyMerged(term_upper_id)
            transition_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged()
            Jʹ = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Jʹʹ = Triangular(L, S)
            Jʹʹʹ = Triangular(L, S)
            Kʹ = Intersection(Triangular(J, Jʹ), Triangular(Jʹʹ, Jʹʹʹ))
            Qʹ = Projection(Kʹ)
            Kr = Intersection(FromTo(0, 2), Triangular(K, Kʹ), Triangular(L, L))
            Qr = Intersection(Projection(Kr), Qʹ - Q)

        if self.relaxation_a_frame is None:

            def _r_a_1(transition_id, J, Jʹ, Jʹʹ, Jʹʹʹ, K, Kʹ, Kr, Q, Qʹ, Qr, L, Lu, S):
                tr = self.transition_registry.transitions[str(transition_id)]
                common = (
                    n_proj(L)
                    * tr.einstein_b_lu
                    * sqrt(n_proj(1, K, Kʹ, Kr))
                    * m1p(1 + Lu - S + J + Qʹ)
                    * wigner_6j(L, L, Kr, 1, 1, Lu)
                    * wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr)
                    * 0.5
                )
                term1 = (
                    delta(J, Jʹʹ)
                    * sqrt(n_proj(Jʹ, Jʹʹʹ))
                    * wigner_6j(L, L, Kr, Jʹʹʹ, Jʹ, S)
                    * wigner_6j(K, Kʹ, Kr, Jʹʹʹ, Jʹ, J)
                )
                term2 = (
                    delta(Jʹ, Jʹʹʹ)
                    * sqrt(n_proj(J, Jʹʹ))
                    * m1p(Jʹʹ - Jʹ + K + Kʹ + Kr)
                    * wigner_6j(L, L, Kr, Jʹʹ, J, S)
                    * wigner_6j(K, Kʹ, Kr, Jʹʹ, J, Jʹ)
                )
                return common * (term1 + term2)

            self.relaxation_a_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_upper(),
                    RelaxationASumLimits(),
                )
                .register_multiplication(_r_a_1, elementwise=True)
                .to_coefficient()
            )

        relaxation_a_frame = self.relaxation_a_frame.copy()
        relaxation_a_frame.register_multiplication(
            lambda transition_id, Kr, Qr: radiation_tensor.get_from_transition_id(
                transition_id=transition_id, K=Kr, Q=Qr
            ),
            elementwise=True,
        )
        relaxation_a_frame.reduce_partially("term_upper_id", "Lu", "transition_id", "Kr", "Qr")

        self.add_coefficient_for_rho(
            frame=relaxation_a_frame,
            multiply_by=-1,
            term_id="term_id",
            K="Kʹ",
            Q="Qʹ",
            J="Jʹʹ",
            Jʹ="Jʹʹʹ",
        )

    @log_method
    def add_relaxation_s(self, radiation_tensor: RadiationTensor):
        r"""
        Add stimulated emission relaxation :math:`R_S = -r_{s1} \cdot J^{K_r}_{Q_r}`.

        Reference: (LL04 7.38, 7.46c)
        """
        # DEPRECATED: disable_r_s gates only this R_S term (not T_S or the RTE eta_S), so it is not a
        # consistent stimulated-emission switch. Scheduled for removal (warned at config construction).
        if self.disable_r_s:
            return

        class RelaxationSSumLimits(SumLimits):
            r"""
            Summation limits for stimulated emission relaxation :math:`R_S = -r_{s1} \cdot J^{K_r}_{Q_r}`.
            """

            term_id = DummyOrAlreadyMerged()
            L = DummyOrAlreadyMerged(term_id)
            S = DummyOrAlreadyMerged(term_id)
            term_lower_id = DummyOrAlreadyMerged()
            Ll = DummyOrAlreadyMerged(term_lower_id)
            transition_id = DummyOrAlreadyMerged()
            J = DummyOrAlreadyMerged()
            Jʹ = DummyOrAlreadyMerged()
            K = DummyOrAlreadyMerged()
            Q = DummyOrAlreadyMerged()
            Jʹʹ = Triangular(L, S)
            Jʹʹʹ = Triangular(L, S)
            Kʹ = Intersection(Triangular(J, Jʹ), Triangular(Jʹʹ, Jʹʹʹ))
            Qʹ = Projection(Kʹ)
            Kr = Intersection(FromTo(0, 2), Triangular(K, Kʹ), Triangular(L, L))
            Qr = Intersection(Projection(Kr), Qʹ - Q)

        if self.relaxation_s_frame is None:

            def _r_s_1(transition_id, J, Jʹ, Jʹʹ, Jʹʹʹ, K, Kʹ, Kr, Q, Qʹ, Qr, L, Ll, S):
                tr = self.transition_registry.transitions[str(transition_id)]
                common = (
                    n_proj(L)
                    * tr.einstein_b_ul
                    * sqrt(n_proj(1, K, Kʹ, Kr))
                    * m1p(1 + Ll - S + J + Kr + Qʹ)
                    * wigner_6j(L, L, Kr, 1, 1, Ll)
                    * wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr)
                    * 0.5
                )
                term1 = (
                    delta(J, Jʹʹ)
                    * sqrt(n_proj(Jʹ, Jʹʹʹ))
                    * wigner_6j(L, L, Kr, Jʹʹʹ, Jʹ, S)
                    * wigner_6j(K, Kʹ, Kr, Jʹʹʹ, Jʹ, J)
                )
                term2 = (
                    delta(Jʹ, Jʹʹʹ)
                    * sqrt(n_proj(J, Jʹʹ))
                    * m1p(Jʹʹ - Jʹ + K + Kʹ + Kr)
                    * wigner_6j(L, L, Kr, Jʹʹ, J, S)
                    * wigner_6j(K, Kʹ, Kr, Jʹʹ, J, Jʹ)
                )
                return common * (term1 + term2)

            self.relaxation_s_frame = (
                Frame.from_sum_limits(
                    self._base_frame_transitions_lower(),
                    RelaxationSSumLimits(),
                )
                .register_multiplication(_r_s_1, elementwise=True)
                .to_coefficient()
            )

        relaxation_s_frame = self.relaxation_s_frame.copy()
        relaxation_s_frame.register_multiplication(
            lambda transition_id, Kr, Qr: radiation_tensor.get_from_transition_id(
                transition_id=transition_id, K=Kr, Q=Qr
            ),
            elementwise=True,
        )
        relaxation_s_frame.reduce_partially("term_lower_id", "Ll", "transition_id", "Kr", "Qr")

        self.add_coefficient_for_rho(
            frame=relaxation_s_frame,
            multiply_by=-1,
            term_id="term_id",
            K="Kʹ",
            Q="Qʹ",
            J="Jʹʹ",
            Jʹ="Jʹʹʹ",
        )

    @log_method
    def get_solution(self) -> Rho:
        r"""
        Get the solution of the Statistical Equilibrium Equations.

        :return: Rho instance

        The solution is constructed by manual linalg solving, which proved to be a bit more reliable
        than available homogeneous solvers. The idea is simple:

        The matrix equation is :math:`A x = 0`.

        Let :math:`x[0] = 1` (it is always :math:`\rho_0^0` of some kind, so it is least likely to be exactly zero).
        Then we have a non-homogeneous system of equations:

        .. math::

            A[1:] x &= 0

            A[1:, 1:] x[1:] &= -A[1:, 0]

        This system generally behaves well enough for linalg.solve to succeed, but pinv is more robust.
        """
        logging.info("Solving Statistical Equilibrium Equations")
        sol = -np.linalg.pinv(self.matrix_builder.rho_matrix[1:, 1:]) @ self.matrix_builder.rho_matrix[1:, 0:1]
        sol = np.insert(sol, 0, 1.0, 0)
        sol = sol[:, 0]

        # Normalize: Sum sqrt(2J+1) rho00(J, J) = 1
        weights = np.zeros_like(sol)
        for index, weight in zip(self.matrix_builder.trace_indexes, self.matrix_builder.trace_weights):
            weights[index] = weight
        trace = (sol * weights).sum()

        rho_vector = sol / trace

        rho = Rho(terms=list(self.level_registry.terms.values()))
        for index, (term_id, k, q, j, j_prime) in self.matrix_builder.index_to_parameters.items():
            rho.set_from_term_id(term_id=term_id, K=k, Q=q, J=j, Jʹ=j_prime, value=rho_vector[index])

        return rho

    def add_coefficient_for_rho(
        self,
        frame: Frame,
        term_id: str,
        K: str,
        Q: str,
        J: str,
        Jʹ: str,
        multiply_by: Union[complex, float, None] = None,
    ):
        df = frame.frame.copy()
        df = self.add_equation_index(df, term_id="term_id", K="K", Q="Q", J="J", Jʹ="Jʹ", index="index0")
        df = self.add_equation_index(df, term_id=term_id, K=K, Q=Q, J=J, Jʹ=Jʹ, index="index1")
        if multiply_by is not None:
            df["coefficient"] = df["coefficient"] * multiply_by
        self.matrix_builder.add_coefficient_from_df(df)

    def add_equation_index(self, df: pd.DataFrame, term_id: str, K: str, Q: str, J: str, Jʹ: str, index: str):
        """
        A helper function to keep track of which matrix row/column each term in SEE corresponds to.
        Set either index0 or index1 using the provided :math:`K, Q,` ....
        """
        # Vectorized: merge the (term_id, K, Q, J, Jʹ) columns against the precomputed
        # index lookup instead of building a coherence-id string per row.
        sub = df[[term_id, K, Q, J, Jʹ]].rename(columns={term_id: "term_id", K: "K", Q: "Q", J: "J", Jʹ: "Jʹ"})
        merged = sub.merge(self.matrix_builder.param_index_df, on=["term_id", "K", "Q", "J", "Jʹ"], how="left")
        df[index] = merged["index"].to_numpy()
        return df
