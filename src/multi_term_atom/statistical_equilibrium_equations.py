from typing import List

import numpy as np
import pandas as pd
from numpy import pi, sqrt
from tqdm import tqdm

from src.common.constants import atomic_mass_unit_g, kB_erg_Km1
from src.common.functions import energy_cmm1_to_erg, energy_cmm1_to_frequency_hz
from src.common.wigner_3j_6j_9j import wigner_3j, wigner_6j, wigner_9j
from src.engine.functions.decorators import log_method
from src.engine.functions.general import delta, m1p, n_proj
from src.engine.functions.looping import (
    FROMTO,
    INTERSECTION,
    PROJECTION,
    TRIANGULAR,
    VALUE,
)
from src.engine.generators.multiply import multiply
from src.engine.generators.nested_loops import nested_loops
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.rho_matrix_builder import (
    Rho,
    RhoMatrixBuilder,
    Term,
    construct_coherence_id_from_term_id,
)
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import (
    TransitionRegistry,
)


class MultiTermAtomSEE:
    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        disable_r_s: bool = False,
        disable_n: bool = False,
        precompute: bool = True,
    ):
        """
        Statistical Equilibrium Equations within Multi-Term atom model

        Reference: (7.38)

        :param level_registry:  LevelRegistry instance for the multi-term atom under study.
        :param transition_registry:  TransitionRegistry instance for the multi-term atom under study.
        :param disable_r_s:  Whether to disable stimulated emission R_S (7.46c).
        Has to be False if precomputed frames are loaded.
        :param disable_n:  Whether to disable the N kernel (7.41).
        Has to be False if precomputed frames are loaded.
        :param precompute:  Whether to precompute some SEE frames.
        Set to False when loading precomputed frames from file.

        Note on precomputing:
        Some parameters are atom-specific, and some are atmosphere-specific.
        E.g. number of levels, their J values etc are atom-specific, while magnetic field is atmosphere-specific.
        So the idea is to precompute everything atom-specific, so that we can quickly iterate over
        atmosphere-specific parameters for this given atom later.
        Also this means that atom-specific precomputing can be saved to disk, and does not need to be performed
        on each launch.
        """
        self.level_registry: LevelRegistry = level_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.matrix_builder: RhoMatrixBuilder = RhoMatrixBuilder(terms=list(self.level_registry.terms.values()))

        self.disable_r_s = disable_r_s
        self.disable_n = disable_n

        # Precomputed frames:
        self.coherence_decay_df = None
        self.absorption_df = None
        self.emission_df_e = None
        self.emission_df_s = None
        self.relaxation_df_a = None
        self.relaxation_df_e = None
        self.relaxation_df_s = None
        if precompute:
            self.precompute_all_equations()
        else:
            assert not disable_n
            assert not disable_r_s

    @log_method
    def precompute_all_equations(self):
        """
        Precompute all frames (instead of loading from a file).
        """
        coherence_decay_dfs = []
        absorption_dfs = []
        emission_dfs_e = []
        emission_dfs_s = []
        relaxation_dfs_a = []
        relaxation_dfs_e = []
        relaxation_dfs_s = []
        for term in tqdm(self.level_registry.terms.values(), leave=False):
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                coherence_decay_dfs.extend(self.precompute_coherence_decay(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ))
                absorption_dfs.extend(self.precompute_absorption(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ))
                emission_dfs_e_, emission_dfs_s_ = self.precompute_emission(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ)
                emission_dfs_e.extend(emission_dfs_e_)
                emission_dfs_s.extend(emission_dfs_s_)
                relaxation_dfs_a_, relaxation_dfs_e_, relaxation_dfs_s_ = self.precompute_relaxation(
                    term=term, K=K, Q=Q, J=J, Jʹ=Jʹ
                )
                relaxation_dfs_a.extend(relaxation_dfs_a_)
                relaxation_dfs_e.extend(relaxation_dfs_e_)
                relaxation_dfs_s.extend(relaxation_dfs_s_)

        self.coherence_decay_df = self.concat_and_finalize_precomputed_dfs(
            coherence_decay_dfs, value_columns=["n_0", "n_1"]
        )
        self.absorption_df = self.concat_and_finalize_precomputed_dfs(absorption_dfs, value_columns=["t_a_1"])
        self.emission_df_e = self.concat_and_finalize_precomputed_dfs(emission_dfs_e, value_columns=["coefficient"])
        self.emission_df_s = self.concat_and_finalize_precomputed_dfs(emission_dfs_s, value_columns=["t_s_1"])
        self.relaxation_df_a = self.concat_and_finalize_precomputed_dfs(relaxation_dfs_a, value_columns=["r_a_1"])
        self.relaxation_df_e = self.concat_and_finalize_precomputed_dfs(relaxation_dfs_e, value_columns=["r_e_0"])
        self.relaxation_df_s = self.concat_and_finalize_precomputed_dfs(relaxation_dfs_s, value_columns=["r_s_1"])

    @log_method
    def fill_all_equations(
        self,
        atmosphere_parameters: AtmosphereParameters,
        radiation_tensor_in_magnetic_frame: RadiationTensor,
    ):
        """
        Loop through all equations to construct the complete system of equations for rho.
        Reference: (7.38)

        :param atmosphere_parameters:  AtmosphereParameters instance carrying the magnetic field and other variables.
        :param radiation_tensor_in_magnetic_frame:  RadiationTensor instance

        Note:
        Calling this function multiple times with different atmosphere_parameters of J tensors is safe,
        it will build the system of equations from scratch each time.
        """

        self.matrix_builder.reset_matrix()
        self.add_precomputed_coherence_decay(self.coherence_decay_df, atmosphere_parameters=atmosphere_parameters)
        self.add_precomputed_absorption(self.absorption_df, radiation_tensor=radiation_tensor_in_magnetic_frame)
        self.add_precomputed_emission(
            self.emission_df_e, self.emission_df_s, radiation_tensor=radiation_tensor_in_magnetic_frame
        )
        self.add_precomputed_relaxation(
            self.relaxation_df_a,
            self.relaxation_df_e,
            self.relaxation_df_s,
            radiation_tensor=radiation_tensor_in_magnetic_frame,
        )

    def precompute_coherence_decay(self, term: Term, K: int, Q: int, J: float, Jʹ: float):
        """
        Precompute all coherence decay N kernel parameters n_0 and n_1:
        N = n_0 + n_1 * nu_larmor

        Reference: (7.38)
        """
        dfs = []
        for Jʹʹ, Jʹʹʹ, Kʹ, Qʹ in nested_loops(
            Jʹʹ=INTERSECTION(TRIANGULAR(term.L, term.S)),
            Jʹʹʹ=INTERSECTION(TRIANGULAR(term.L, term.S)),
            Kʹ=INTERSECTION(TRIANGULAR(K, 1), TRIANGULAR("Jʹʹ", "Jʹʹʹ")),
            Qʹ=INTERSECTION(PROJECTION("Kʹ"), VALUE(Q)),
        ):
            df = self.precompute_n(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            df = self.add_equation_index1(df, term_id="term_id", K="Kʹ", Q="Qʹ", J="Jʹʹ", Jʹ="Jʹʹʹ")
            dfs.append(df)
        return dfs

    @log_method
    def add_precomputed_coherence_decay(self, df, atmosphere_parameters: AtmosphereParameters):
        """
        Add the coherence decay N kernel using precomputed n_0 and n_1:
        N = n_0 + n_1 * nu_larmor

        Reference: (7.38)
        """
        df["coefficient"] = -2 * pi * 1j * (df.n_0 + df.n_1 * atmosphere_parameters.nu_larmor)
        self.matrix_builder.add_coefficient_from_df(df)

    def precompute_absorption(self, term: Term, K: int, Q: int, J: float, Jʹ: float):
        """
        Precompute all Absorption parameters t_a_1:
        T_a = t_a_1 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr)

        Reference: (7.38)
        """
        dfs = []
        for term_lower in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term, term_lower=term_lower):
                continue
            for Jl, Jʹl, Kl, Ql in nested_loops(
                Jl=INTERSECTION(TRIANGULAR(term_lower.L, term_lower.S), TRIANGULAR(J, 1)),
                Jʹl=INTERSECTION(TRIANGULAR(term_lower.L, term_lower.S), TRIANGULAR(Jʹ, 1)),
                Kl=TRIANGULAR("Jl", "Jʹl"),
                Ql=PROJECTION("Kl"),
            ):
                t_a_dfs = self.precompute_t_a(
                    term=term, K=K, Q=Q, J=J, Jʹ=Jʹ, term_lower=term_lower, Kl=Kl, Ql=Ql, Jl=Jl, Jʹl=Jʹl
                )
                t_a_dfs = [
                    self.add_equation_index1(df, term_id="term_lower_id", K="Kl", Q="Ql", J="Jl", Jʹ="Jʹl")
                    for df in t_a_dfs
                ]
                dfs.extend(t_a_dfs)
        return dfs

    @log_method
    def add_precomputed_absorption(self, df, radiation_tensor: RadiationTensor):
        """
        Add the Absorption using precomputed parameter t_a_1:
        T_a = t_a_1 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr)

        Reference: (7.38)
        """
        df = df.merge(
            radiation_tensor.df.rename(
                columns={"K": "Kr", "Q": "Qr"},
            ),
            how="inner",
        )

        df["coefficient"] = df.t_a_1 * df.radiation_tensor
        self.matrix_builder.add_coefficient_from_df(df)

    def precompute_emission(self, term: Term, K: int, Q: int, J: float, Jʹ: float):
        """
        Precompute all Emission parameters coefficient, t_s_1:
        T_e = coefficient
        T_s = t_s_1 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr)

        Reference: (7.38)
        """
        dfs_e = []
        dfs_s = []
        for term_upper in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term_upper, term_lower=term):
                continue

            for Ju, Jʹu, Ku, Qu in nested_loops(
                Ju=INTERSECTION(TRIANGULAR(term_upper.L, term_upper.S), TRIANGULAR(J, 1)),
                Jʹu=INTERSECTION(TRIANGULAR(term_upper.L, term_upper.S), TRIANGULAR(Jʹ, 1)),
                Ku=TRIANGULAR("Ju", "Jʹu"),
                Qu=PROJECTION("Ku"),
            ):
                t_e_dfs = self.precompute_t_e(
                    term=term, K=K, Q=Q, J=J, Jʹ=Jʹ, term_upper=term_upper, Ku=Ku, Qu=Qu, Ju=Ju, Jʹu=Jʹu
                )
                t_s_dfs = self.precompute_t_s(
                    term=term, K=K, Q=Q, J=J, Jʹ=Jʹ, term_upper=term_upper, Ku=Ku, Qu=Qu, Ju=Ju, Jʹu=Jʹu
                )

                t_e_dfs = [
                    self.add_equation_index1(df, term_id="term_upper_id", K="Ku", Q="Qu", J="Ju", Jʹ="Jʹu")
                    for df in t_e_dfs
                ]
                t_s_dfs = [
                    self.add_equation_index1(df, term_id="term_upper_id", K="Ku", Q="Qu", J="Ju", Jʹ="Jʹu")
                    for df in t_s_dfs
                ]
                dfs_e.extend(t_e_dfs)
                dfs_s.extend(t_s_dfs)
        return dfs_e, dfs_s

    @log_method
    def add_precomputed_emission(self, df_e, df_s, radiation_tensor: RadiationTensor):
        """
        Add the Emission using precomputed parameter t_a_1:
        T_e = coefficient
        T_s = t_s_1 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr)

        Reference: (7.38)
        """
        self.matrix_builder.add_coefficient_from_df(df_e)

        df_s = df_s.merge(
            radiation_tensor.df.rename(
                columns={"K": "Kr", "Q": "Qr"},
            ),
            how="inner",
        )
        df_s["coefficient"] = df_s.t_s_1 * df_s.radiation_tensor
        self.matrix_builder.add_coefficient_from_df(df_s)

    def precompute_relaxation(self, term: Term, K: int, Q: int, J: float, Jʹ: float):
        """
        Precompute all relaxation parameters r_a_1, r_e_0, r_s_1:
        R_A = - r_a_1 * radiation_tensor
        R_E = - r_e_0
        R_S = - r_s_1 * radiation_tensor

        Reference: (7.38)
        """
        dfs_a = []
        dfs_e = []
        dfs_s = []

        # Relaxation from selected coherence
        for Jʹʹ, Jʹʹʹ, Kʹ, Qʹ in nested_loops(
            Jʹʹ=TRIANGULAR(term.L, term.S),
            Jʹʹʹ=TRIANGULAR(term.L, term.S),
            Kʹ=INTERSECTION(TRIANGULAR(J, Jʹ), TRIANGULAR("Jʹʹ", "Jʹʹʹ")),
            Qʹ=PROJECTION("Kʹ"),
        ):
            r_a_dfs = self.precompute_r_a(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            r_e_dfs = self.precompute_r_e(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            r_s_dfs = self.precompute_r_s(term=term, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)

            r_a_dfs = [
                self.add_equation_index1(df, term_id="term_id", K="Kʹ", Q="Qʹ", J="Jʹʹ", Jʹ="Jʹʹʹ") for df in r_a_dfs
            ]
            r_e_dfs = [
                self.add_equation_index1(df, term_id="term_id", K="Kʹ", Q="Qʹ", J="Jʹʹ", Jʹ="Jʹʹʹ") for df in r_e_dfs
            ]
            r_s_dfs = [
                self.add_equation_index1(df, term_id="term_id", K="Kʹ", Q="Qʹ", J="Jʹʹ", Jʹ="Jʹʹʹ") for df in r_s_dfs
            ]
            dfs_a.extend(r_a_dfs)
            dfs_e.extend(r_e_dfs)
            dfs_s.extend(r_s_dfs)

        return dfs_a, dfs_e, dfs_s

    @log_method
    def add_precomputed_relaxation(self, df_a, df_e, df_s, radiation_tensor):
        """
        Add all relaxation rates using precomputed parameters r_a_1, r_e_0, r_s_1:
        R_A = - r_a_1 * radiation_tensor
        R_E = - r_e_0
        R_S = - r_s_1 * radiation_tensor

        Reference: (7.38)
        """
        df_a = df_a.merge(
            radiation_tensor.df.rename(
                columns={"K": "Kr", "Q": "Qr"},
            ),
            how="inner",
        )
        df_a["coefficient"] = -df_a.r_a_1 * df_a.radiation_tensor
        self.matrix_builder.add_coefficient_from_df(df_a)

        df_e["coefficient"] = -df_e.r_e_0
        self.matrix_builder.add_coefficient_from_df(df_e)

        df_s = df_s.merge(
            radiation_tensor.df.rename(
                columns={"K": "Kr", "Q": "Qr"},
            ),
            how="inner",
        )
        df_s["coefficient"] = -df_s.r_s_1 * df_s.radiation_tensor
        self.matrix_builder.add_coefficient_from_df(df_s)

    def precompute_r_a(
        self, term: Term, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float
    ):
        """
        Compute r_a_1

        Reference: (7.46a)
        """
        L = term.L
        S = term.S

        dfs = []
        for term_upper in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term_upper, term_lower=term):
                continue

            transition = self.transition_registry.get_transition(term_upper=term_upper, term_lower=term)
            Lu = term_upper.L

            for Kr, Qr in nested_loops(
                Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(K, Kʹ), TRIANGULAR(L, L)),
                Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qʹ - Q)),
            ):
                r_a_1 = multiply(
                    lambda: n_proj(L) * transition.einstein_b_lu,
                    lambda: sqrt(n_proj(1, K, Kʹ, Kr)),
                    lambda: m1p(1 + Lu - S + J + Qʹ),
                    lambda: wigner_6j(L, L, Kr, 1, 1, Lu) * wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr),
                    lambda: 0.5,
                    lambda: delta(J, Jʹʹ),
                    lambda: sqrt(n_proj(Jʹ, Jʹʹʹ)),
                    lambda: wigner_6j(L, L, Kr, Jʹʹʹ, Jʹ, S),
                    lambda: wigner_6j(K, Kʹ, Kr, Jʹʹʹ, Jʹ, J),
                    is_scalar=True,
                )
                r_a_1 += (
                    n_proj(L)
                    * transition.einstein_b_lu
                    * sqrt(n_proj(1, K, Kʹ, Kr))
                    * m1p(1 + Lu - S + J + Qʹ)
                    * wigner_6j(L, L, Kr, 1, 1, Lu)
                    * wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr)
                    * 0.5
                    * delta(Jʹ, Jʹʹʹ)
                    * sqrt(n_proj(J, Jʹʹ))
                    * m1p(Jʹʹ - Jʹ + K + Kʹ + Kr)
                    * wigner_6j(L, L, Kr, Jʹʹ, J, S)
                    * wigner_6j(K, Kʹ, Kr, Jʹʹ, J, Jʹ)
                )
                dfs.append(
                    pd.DataFrame(
                        {
                            "term_id": [term.term_id],
                            "K": [K],
                            "Q": [Q],
                            "J": [J],
                            "Jʹ": [Jʹ],
                            "term_upper_id": [term_upper.term_id],
                            "transition_id": [transition.transition_id],
                            "Kʹ": [Kʹ],
                            "Qʹ": [Qʹ],
                            "Jʹʹ": [Jʹʹ],
                            "Jʹʹʹ": [Jʹʹʹ],
                            "Kr": [Kr],
                            "Qr": [Qr],
                            "r_a_1": [r_a_1],
                        }
                    )
                )

        return dfs

    def precompute_r_e(
        self, term: Term, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float
    ):
        """
        Compute r_e_0

        Reference: (7.46b)
        """
        dfs = []
        for term_lower in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term, term_lower=term_lower):
                continue
            transition = self.transition_registry.get_transition(term_upper=term, term_lower=term_lower)
            r_e_0 = delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ) * transition.einstein_a_ul
            dfs.append(
                pd.DataFrame(
                    {
                        "term_id": [term.term_id],
                        "K": [K],
                        "Q": [Q],
                        "J": [J],
                        "Jʹ": [Jʹ],
                        "term_lower_id": [term_lower.term_id],
                        "transition_id": [transition.transition_id],
                        "Kʹ": [Kʹ],
                        "Qʹ": [Qʹ],
                        "Jʹʹ": [Jʹʹ],
                        "Jʹʹʹ": [Jʹʹʹ],
                        "r_e_0": [r_e_0],
                    }
                )
            )
        return dfs

    def precompute_r_s(
        self, term: Term, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float
    ):
        """
        Compute r_s_1

        Reference: (7.46c)
        """
        L = term.L
        S = term.S
        dfs = []
        for term_lower in self.level_registry.terms.values():
            if not self.transition_registry.is_transition_registered(term_upper=term, term_lower=term_lower):
                continue
            transition = self.transition_registry.get_transition(term_upper=term, term_lower=term_lower)
            Ll = term_lower.L
            for Kr, Qr in nested_loops(
                Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(K, Kʹ), TRIANGULAR(L, L)),
                Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qʹ - Q)),
            ):
                r_s_1 = multiply(
                    lambda: n_proj(L) * transition.einstein_b_ul,
                    lambda: sqrt(n_proj(1, K, Kʹ, Kr)),
                    lambda: m1p(1 + Ll - S + J + Kr + Qʹ),
                    lambda: wigner_6j(L, L, Kr, 1, 1, Ll),
                    lambda: wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr),
                    lambda: 0.5,
                    lambda: delta(J, Jʹʹ),
                    lambda: sqrt(n_proj(Jʹ, Jʹʹʹ)),
                    lambda: wigner_6j(L, L, Kr, Jʹʹʹ, Jʹ, S),
                    lambda: wigner_6j(K, Kʹ, Kr, Jʹʹʹ, Jʹ, J),
                    is_scalar=True,
                )

                r_s_1 += multiply(
                    lambda: n_proj(L) * transition.einstein_b_ul,
                    lambda: sqrt(n_proj(1, K, Kʹ, Kr)),
                    lambda: m1p(1 + Ll - S + J + Kr + Qʹ),
                    lambda: wigner_6j(L, L, Kr, 1, 1, Ll),
                    lambda: wigner_3j(K, Kʹ, Kr, Q, -Qʹ, Qr),
                    lambda: 0.5,
                    lambda: delta(Jʹ, Jʹʹʹ) * sqrt(n_proj(J, Jʹʹ)),
                    lambda: m1p(Jʹʹ - Jʹ + K + Kʹ + Kr),
                    lambda: wigner_6j(L, L, Kr, Jʹʹ, J, S),
                    lambda: wigner_6j(K, Kʹ, Kr, Jʹʹ, J, Jʹ),
                    is_scalar=True,
                )
                if self.disable_r_s:
                    r_s_1 = r_s_1 * 0

                dfs.append(
                    pd.DataFrame(
                        {
                            "term_id": [term.term_id],
                            "K": [K],
                            "Q": [Q],
                            "J": [J],
                            "Jʹ": [Jʹ],
                            "term_lower_id": [term_lower.term_id],
                            "transition_id": [transition.transition_id],
                            "Kʹ": [Kʹ],
                            "Qʹ": [Qʹ],
                            "Jʹʹ": [Jʹʹ],
                            "Jʹʹʹ": [Jʹʹʹ],
                            "Kr": [Kr],
                            "Qr": [Qr],
                            "r_s_1": [r_s_1],
                        }
                    )
                )

        return dfs

    @staticmethod
    def gamma(term: Term, J: float, Jʹ: float):
        """
        Compute Gamma which is in the coherence relaxation kernel N.

        Reference: (7.42)
        """

        S = term.S
        L = term.L
        result = delta(J, Jʹ) * sqrt(J * (J + 1) * (2 * J + 1))
        result += (
            m1p(1 + L + S + J)
            * sqrt((2 * J + 1) * (2 * Jʹ + 1) * S * (S + 1) * (2 * S + 1))
            * wigner_6j(J, Jʹ, 1, S, S, L)
        )
        return result

    def precompute_n(self, term: Term, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float):
        """
        Precompute coherence relaxation parameters n_0, n_1:
        N = n_0 + n_1 * nu_larmor

        Reference: (7.41)
        """

        level = self.level_registry.get_level(term=term, J=J)
        level_prime = self.level_registry.get_level(term=term, J=Jʹ)
        nu = energy_cmm1_to_frequency_hz(level.energy_cmm1 - level_prime.energy_cmm1)

        n_0 = delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ) * nu

        n_1 = (
            delta(Q, Qʹ)
            * m1p(J + Jʹ - Q)
            * sqrt((2 * K + 1) * (2 * Kʹ + 1))
            * wigner_3j(K, Kʹ, 1, -Q, Q, 0)
            * (
                delta(Jʹ, Jʹʹʹ) * self.gamma(term=term, J=J, Jʹ=Jʹʹ) * wigner_6j(K, Kʹ, 1, Jʹʹ, J, Jʹ)
                + delta(J, Jʹʹ) * m1p(K - Kʹ) * self.gamma(term=term, J=Jʹʹʹ, Jʹ=Jʹ) * wigner_6j(K, Kʹ, 1, Jʹʹʹ, Jʹ, J)
            )
        )
        if self.disable_n:
            n_0 = n_0 * 0
            n_1 = n_1 * 0

        df = pd.DataFrame(
            {
                "term_id": [term.term_id],
                "K": [K],
                "Q": [Q],
                "J": [J],
                "Jʹ": [Jʹ],
                "Kʹ": [Kʹ],
                "Qʹ": [Qʹ],
                "Jʹʹ": [Jʹʹ],
                "Jʹʹʹ": [Jʹʹʹ],
                "n_0": [n_0],
                "n_1": [n_1],
            }
        )
        return df

    def precompute_t_a(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        term_lower: Term,
        Kl: int,
        Ql: int,
        Jl: float,
        Jʹl: float,
    ):
        """
        Precompute t_a_1

        Reference: (7.45a)
        """
        S = term.S
        L = term.L
        Ll = term_lower.L

        transition = self.transition_registry.get_transition(term_upper=term, term_lower=term_lower)
        dfs = []
        for Kr, Qr in nested_loops(
            Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(K, Kl)), Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Ql - Q))
        ):
            t_a_1 = multiply(
                lambda: n_proj(Ll) * transition.einstein_b_lu,
                lambda: sqrt(n_proj(1, J, Jʹ, Jl, Jʹl, K, Kl, Kr)),
                lambda: m1p(Kl + Ql + Jʹl - Jl),
                lambda: wigner_9j(J, Jl, 1, Jʹ, Jʹl, 1, K, Kl, Kr),
                lambda: wigner_6j(L, Ll, 1, Jl, J, S),
                lambda: wigner_6j(L, Ll, 1, Jʹl, Jʹ, S),
                lambda: wigner_3j(K, Kl, Kr, -Q, Ql, -Qr),
                is_scalar=True,
            )
            dfs.append(
                pd.DataFrame(
                    {
                        "term_id": [term.term_id],
                        "K": [K],
                        "Q": [Q],
                        "J": [J],
                        "Jʹ": [Jʹ],
                        "term_lower_id": [term_lower.term_id],
                        "transition_id": [transition.transition_id],
                        "Kl": [Kl],
                        "Ql": [Ql],
                        "Jl": [Jl],
                        "Jʹl": [Jʹl],
                        "Kr": [Kr],
                        "Qr": [Qr],
                        "t_a_1": [t_a_1],
                    }
                )
            )

        return dfs

    def precompute_t_e(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        term_upper: Term,
        Ku: int,
        Qu: int,
        Ju: float,
        Jʹu: float,
    ):
        """
        Precompute Te coefficient

        Reference: (7.45b)
        """

        S = term.S
        L = term.L
        Lu = term_upper.L

        transition = self.transition_registry.get_transition(term_upper=term_upper, term_lower=term)
        assert S == term_upper.S

        coefficient = multiply(
            lambda: delta(S, term_upper.S) * delta(K, Ku) * delta(Q, Qu),
            lambda: (2 * Lu + 1) * transition.einstein_a_ul,
            lambda: sqrt(n_proj(J, Jʹ, Ju, Jʹu)),
            lambda: m1p(1 + K + Jʹ + Jʹu),
            lambda: wigner_6j(J, Jʹ, K, Jʹu, Ju, 1),
            lambda: wigner_6j(Lu, L, 1, J, Ju, S),
            lambda: wigner_6j(Lu, L, 1, Jʹ, Jʹu, S),
            is_scalar=True,
        )

        return [
            pd.DataFrame(
                {
                    "term_id": [term.term_id],
                    "K": [K],
                    "Q": [Q],
                    "J": [J],
                    "Jʹ": [Jʹ],
                    "term_upper_id": [term_upper.term_id],
                    "transition_id": [transition.transition_id],
                    "Ku": [Ku],
                    "Qu": [Qu],
                    "Ju": [Ju],
                    "Jʹu": [Jʹu],
                    "coefficient": [coefficient],
                }
            )
        ]

    def precompute_t_s(
        self,
        term: Term,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        term_upper: Term,
        Ku: int,
        Qu: int,
        Ju: float,
        Jʹu: float,
    ):
        """
        Precompute t_s_1

        Reference: (7.45c)
        """
        S = term.S
        L = term.L
        Lu = term_upper.L

        transition = self.transition_registry.get_transition(term_upper=term_upper, term_lower=term)
        dfs = []
        for Kr, Qr in nested_loops(
            Kr=INTERSECTION(FROMTO(0, 2), TRIANGULAR(Ku, K)),
            Qr=INTERSECTION(PROJECTION("Kr"), VALUE(Qu - Q)),
        ):
            t_s_1 = multiply(
                lambda: n_proj(Lu) * transition.einstein_b_ul,
                lambda: sqrt(n_proj(J, Jʹ, Ju, Jʹu, K, Ku, Kr)),
                lambda: m1p(Kr + Ku + Qu + Jʹu - Ju),
                lambda: wigner_9j(J, Ju, 1, Jʹ, Jʹu, 1, K, Ku, Kr),
                lambda: wigner_6j(Lu, L, 1, J, Ju, S),
                lambda: wigner_6j(Lu, L, 1, Jʹ, Jʹu, S),
                lambda: wigner_3j(K, Ku, Kr, -Q, Qu, -Qr),
                is_scalar=True,
            )

            dfs.append(
                pd.DataFrame(
                    {
                        "term_id": [term.term_id],
                        "K": [K],
                        "Q": [Q],
                        "J": [J],
                        "Jʹ": [Jʹ],
                        "term_upper_id": [term_upper.term_id],
                        "transition_id": [transition.transition_id],
                        "Ku": [Ku],
                        "Qu": [Qu],
                        "Ju": [Ju],
                        "Jʹu": [Jʹu],
                        "t_s_1": [t_s_1],
                        "Kr": [Kr],
                        "Qr": [Qr],
                    }
                )
            )

        return dfs

    @log_method
    def get_solution(self) -> Rho:
        """
        Get the solution of the Statistical Equilibrium Equations.

        :return: Rho instance

        The solution is constructed by manual linalg solving, which proved to be a bit more reliable
        than available homogeneous solvers. The idea is simple:

        The matrix equation is A x = 0.
        let x[0] = 1 (it is always rho_0_0 of some kind, so it is least likely to be exactly zero)
        Then we have a non-homogeneous system of equations:
        A[1:] x = 0
        A[1:, 1:] x[1:] = -A[1:, 0]
        This system generally behaves well enough for linalg.solve to succeed.
        """
        sol = np.linalg.solve(
            self.matrix_builder.rho_matrix[:, 1:, 1:],
            -self.matrix_builder.rho_matrix[:, 1:, 0:1],
        )
        sol = np.insert(sol, 0, 1.0, 1)
        sol = sol[:, :, 0]

        # Normalize the solution:
        # Sum sqrt(2J+1) rho00(J, J) = 1
        weights = np.zeros_like(sol)
        for index, weight in zip(self.matrix_builder.trace_indexes, self.matrix_builder.trace_weights):
            weights[:, index] = weight
        trace = (sol * weights).sum(axis=1, keepdims=True)

        solution_vector = sol / trace

        # Fill out the Rho instance
        rho = Rho(terms=list(self.level_registry.terms.values()))
        for index, (term_id, k, q, j, j_prime) in self.matrix_builder.index_to_parameters.items():
            rho.set_from_term_id(term_id=term_id, K=k, Q=q, J=j, Jʹ=j_prime, value=solution_vector[:, index])

        return rho

    def concat_and_finalize_precomputed_dfs(self, dfs: List[pd.DataFrame], value_columns: List[str]) -> pd.DataFrame:
        """
        A helper function to finalize the precomputed frames.

        :param dfs: list of dataframes.
        :param value_columns: Todo: remove this parameter.
        :return: finalized precomputed frame.
        """
        assert len(dfs) > 0, "Empty precomputed of dataframe"
        df = pd.concat(dfs, ignore_index=True)
        df = self.add_equation_index0(df=df, term_id="term_id", K="K", Q="Q", J="J", Jʹ="Jʹ")
        return df

    def add_equation_index0(self, df: pd.DataFrame, term_id: str, K: str, Q: str, J: str, Jʹ: str):
        """
        A helper function to keep track of which matrix row/column each term in SEE corresponds to.
        Set index0 (row).
        """
        return self._add_equation_index(df=df, term_id=term_id, K=K, Q=Q, J=J, Jʹ=Jʹ, index="index0")

    def add_equation_index1(self, df: pd.DataFrame, term_id: str, K: str, Q: str, J: str, Jʹ: str):
        """
        A helper function to keep track of which matrix row/column each term in SEE corresponds to.
        Set index1 (column).
        """
        return self._add_equation_index(df=df, term_id=term_id, K=K, Q=Q, J=J, Jʹ=Jʹ, index="index1")

    def _add_equation_index(self, df: pd.DataFrame, term_id: str, K: str, Q: str, J: str, Jʹ: str, index: str):
        """
        A helper function to keep track of which matrix row/column each term in SEE corresponds to.
        Set either index0 or index1 using the provided K, Q, ... .
        See src.multi_term_atom.object.rho_matrix_builder.RhoMatrixBuilder.add_coefficient_from_df for reference.
        """
        df[index] = df.apply(
            lambda row: self.matrix_builder.coherence_id_to_index[
                construct_coherence_id_from_term_id(term_id=row[term_id], K=row[K], Q=row[Q], J=row[J], Jʹ=row[Jʹ])
            ],
            axis=1,
        )
        return df


class MultiTermAtomSEELTE:
    def __init__(
        self,
        level_registry: LevelRegistry,
        atomic_mass_amu: float,
    ):
        """
        Statistical Equilibrium Equations within Multi-Term atom model - an LTE implementation.
        This class will always output an LTE-distributed Rho tensor.

        TODO: need reference.

        :param level_registry:  LevelRegistry instance for the multi-term atom under study.
        This is needed to be able to use SEELTE directly in nonLTE Radiative Transfer Equations.
        """
        self.level_registry = level_registry
        self.atomic_mass_amu = atomic_mass_amu
        self.matrix_builder: RhoMatrixBuilder = RhoMatrixBuilder(terms=list(self.level_registry.terms.values()))

    @log_method
    def temperature_from_delta_v(self, delta_v_thermal_cm_sm1: float) -> float:
        """
        TODO: DEPRECATE IN FAVOR OF T IN ATMOSPHERE_PARAMETERS
        delta_v = sqrt(2 k_B T / m)
        """
        m_g = self.atomic_mass_amu * atomic_mass_unit_g
        return m_g * delta_v_thermal_cm_sm1**2 / (2 * kB_erg_Km1)

    @log_method
    def get_solution(self, atmosphere_parameters: AtmosphereParameters) -> Rho:
        """
        Return LTE Rho solution.

        TODO: need reference.
        """
        # TODO: DEPRECATE IN FAVOR OF T IN ATMOSPHERE_PARAMETERS
        T = self.temperature_from_delta_v(atmosphere_parameters.delta_v_thermal_cm_sm1)

        rho = Rho(terms=list(self.level_registry.terms.values()))
        for index, (term_id, k, q, j, j_prime) in self.matrix_builder.index_to_parameters.items():
            rho.set_from_term_id(term_id=term_id, K=k, Q=q, J=j, Jʹ=j_prime, value=0.0)

        for term in self.level_registry.terms.values():
            levels = term.levels

            # Boltzmann weights
            weights = {}
            for level in levels:
                J = level.J
                E_erg = energy_cmm1_to_erg(level.energy_cmm1)
                weights[level] = (2 * J + 1) * np.exp(-E_erg / (kB_erg_Km1 * T))

            Z = sum(weights.values())
            if Z == 0:
                raise ValueError("LTE partition sum is zero (Too high energies or temperature too low?)")

            for level, w in weights.items():
                J = level.J
                rho_00 = np.sqrt(2 * J + 1) * w / Z

                rho.set_from_term_id(
                    term_id=term.term_id,
                    K=0,
                    Q=0,
                    J=J,
                    Jʹ=J,
                    value=rho_00,
                )

        return rho
