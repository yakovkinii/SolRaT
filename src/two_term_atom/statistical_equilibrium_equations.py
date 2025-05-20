import logging
from typing import List

import numpy as np
import pandas as pd
from numpy import pi, sqrt
from tqdm import tqdm

from src.core.engine.functions.decorators import log_method
from src.core.engine.functions.general import delta, m1p, n_proj
from src.core.engine.functions.looping import FROMTO, INTERSECTION, PROJECTION, TRIANGULAR, VALUE
from src.core.engine.generators.multiply import multiply
from src.core.engine.generators.nested_loops import nested_loops
from src.core.physics.functions import energy_cmm1_to_frequency_hz
from src.core.physics.wigner_3j_6j_9j import wigner_3j, wigner_6j, wigner_9j
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.object.rho_matrix_builder import (
    Level,
    Rho,
    RhoMatrixBuilder,
    construct_coherence_id_from_level_id,
)
from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class TwoTermAtomSEE:
    def __init__(
        self,
        term_registry: TermRegistry,
        transition_registry: TransitionRegistry,
        # atmosphere_parameters: AtmosphereParameters,
        # radiation_tensor: RadiationTensor,
        # n_frequencies: int = 1,
        disable_r_s: bool = False,
        disable_n: bool = False,
        precompute: bool = True,
    ):
        n_frequencies = 1
        self.term_registry: TermRegistry = term_registry
        self.transition_registry: TransitionRegistry = transition_registry
        self.matrix_builder: RhoMatrixBuilder = RhoMatrixBuilder(
            levels=list(self.term_registry.levels.values()), n_frequencies=n_frequencies
        )
        # self.atmosphere_parameters: AtmosphereParameters = atmosphere_parameters
        # self.radiation_tensor: RadiationTensor = radiation_tensor
        self.disable_r_s = disable_r_s
        self.disable_n = disable_n

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

    def concat_and_finalize_precomputed_dfs(self, dfs: List[pd.DataFrame], value_columns: List[str]) -> pd.DataFrame:
        assert len(dfs) > 0, "Empty precomputed of dataframe"
        df = pd.concat(dfs, ignore_index=True)
        # df = df.loc[~((df.n_1 == 0) & (df.n_2 == 0)), :].copy()
        for col in value_columns:
            df[col] = df[col].apply(lambda x: np.array([x]) if np.isscalar(x) else x)
        df = self.add_equation_index0(
            df=df,
            level_id="level_id",
            K="K",
            Q="Q",
            J="J",
            Jʹ="Jʹ",
        )
        return df

    @log_method
    def precompute_all_equations(self):
        coherence_decay_dfs = []
        absorption_dfs = []
        emission_dfs_e = []
        emission_dfs_s = []
        relaxation_dfs_a = []
        relaxation_dfs_e = []
        relaxation_dfs_s = []
        for level in tqdm(self.term_registry.levels.values(), leave=False):
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(level.L, level.S),
                Jʹ=TRIANGULAR(level.L, level.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                coherence_decay_dfs.extend(self.precompute_coherence_decay(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ))
                absorption_dfs.extend(self.precompute_absorption(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ))
                emission_dfs_e_, emission_dfs_s_ = self.precompute_emission(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)
                emission_dfs_e.extend(emission_dfs_e_)
                emission_dfs_s.extend(emission_dfs_s_)
                relaxation_dfs_a_, relaxation_dfs_e_, relaxation_dfs_s_ = self.precompute_relaxation(
                    level=level, K=K, Q=Q, J=J, Jʹ=Jʹ
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
    def add_all_equations(
        self,
        atmosphere_parameters: AtmosphereParameters,
        radiation_tensor_in_magnetic_frame: RadiationTensor,
    ):
        """
        Loops through all equations.

        Reference: (7.38)
        """
        self.matrix_builder.reset_matrix()
        self.add_precomputed_coherence_decay(self.coherence_decay_df, atmosphere_parameters=atmosphere_parameters)
        self.add_precomputed_absorption(self.absorption_df, radiation_tensor=radiation_tensor_in_magnetic_frame)
        self.add_precomputed_emission(
            self.emission_df_e,
            self.emission_df_s,
            radiation_tensor=radiation_tensor_in_magnetic_frame,
        )
        self.add_precomputed_relaxation(
            self.relaxation_df_a,
            self.relaxation_df_e,
            self.relaxation_df_s,
            radiation_tensor=radiation_tensor_in_magnetic_frame,
        )

    def precompute_coherence_decay(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        """
        N = n_0 + n_1 * nu_larmor
        Reference: (7.38)
        """
        dfs = []
        for Jʹʹ, Jʹʹʹ, Kʹ, Qʹ in nested_loops(
            Jʹʹ=INTERSECTION(TRIANGULAR(level.L, level.S)),
            Jʹʹʹ=INTERSECTION(TRIANGULAR(level.L, level.S)),
            Kʹ=INTERSECTION(TRIANGULAR(K, 1), TRIANGULAR("Jʹʹ", "Jʹʹʹ")),
            Qʹ=INTERSECTION(PROJECTION("Kʹ"), VALUE(Q)),
        ):
            df = self.precompute_n(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            df = self.add_equation_index1(df, level_id="level_id", K="Kʹ", Q="Qʹ", J="Jʹʹ", Jʹ="Jʹʹʹ")
            dfs.append(df)
        return dfs

    @log_method
    def add_precomputed_coherence_decay(self, df, atmosphere_parameters: AtmosphereParameters):
        """
        N = n_0 + n_1 * nu_larmor
        Reference: (7.38)
        """
        df["coefficient"] = -2 * pi * 1j * (df.n_0 + df.n_1 * atmosphere_parameters.nu_larmor)
        self.matrix_builder.add_coefficient_from_df(df)

    def precompute_absorption(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        """
        T_a = t_a_1 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr)
        Reference: (7.38)
        """
        # Absorption toward selected coherence
        dfs = []
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level, level_lower=level_lower):
                continue
            for Jl, Jʹl, Kl, Ql in nested_loops(
                Jl=INTERSECTION(TRIANGULAR(level_lower.L, level_lower.S), TRIANGULAR(J, 1)),
                Jʹl=INTERSECTION(TRIANGULAR(level_lower.L, level_lower.S), TRIANGULAR(Jʹ, 1)),
                Kl=TRIANGULAR("Jl", "Jʹl"),
                Ql=PROJECTION("Kl"),
            ):
                t_a_dfs = self.precompute_t_a(
                    level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, level_lower=level_lower, Kl=Kl, Ql=Ql, Jl=Jl, Jʹl=Jʹl
                )
                t_a_dfs = [
                    self.add_equation_index1(df, level_id="level_lower_id", K="Kl", Q="Ql", J="Jl", Jʹ="Jʹl")
                    for df in t_a_dfs
                ]
                dfs.extend(t_a_dfs)
        return dfs

    @log_method
    def add_precomputed_absorption(self, df, radiation_tensor: RadiationTensor):
        df = df.merge(
            radiation_tensor.df.rename(
                columns={"K": "Kr", "Q": "Qr"},
            ),
            how="inner",
        )

        df["coefficient"] = df.t_a_1 * df.radiation_tensor
        self.matrix_builder.add_coefficient_from_df(df)

    def precompute_emission(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        """
        T_e = coefficient
        T_s = t_s_1 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr)

        Reference: (7.38)
        """
        # Emission toward selected coherence
        dfs_e = []
        dfs_s = []
        for level_upper in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level_upper, level_lower=level):
                continue

            for Ju, Jʹu, Ku, Qu in nested_loops(
                Ju=INTERSECTION(TRIANGULAR(level_upper.L, level_upper.S), TRIANGULAR(J, 1)),
                Jʹu=INTERSECTION(TRIANGULAR(level_upper.L, level_upper.S), TRIANGULAR(Jʹ, 1)),
                Ku=TRIANGULAR("Ju", "Jʹu"),
                Qu=PROJECTION("Ku"),
            ):
                t_e_dfs = self.precompute_t_e(
                    level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, level_upper=level_upper, Ku=Ku, Qu=Qu, Ju=Ju, Jʹu=Jʹu
                )
                t_s_dfs = self.precompute_t_s(
                    level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, level_upper=level_upper, Ku=Ku, Qu=Qu, Ju=Ju, Jʹu=Jʹu
                )

                t_e_dfs = [
                    self.add_equation_index1(df, level_id="level_upper_id", K="Ku", Q="Qu", J="Ju", Jʹ="Jʹu")
                    for df in t_e_dfs
                ]
                t_s_dfs = [
                    self.add_equation_index1(df, level_id="level_upper_id", K="Ku", Q="Qu", J="Ju", Jʹ="Jʹu")
                    for df in t_s_dfs
                ]
                dfs_e.extend(t_e_dfs)
                dfs_s.extend(t_s_dfs)
        return dfs_e, dfs_s

    @log_method
    def add_precomputed_emission(self, df_e, df_s, radiation_tensor: RadiationTensor):
        self.matrix_builder.add_coefficient_from_df(df_e)

        df_s = df_s.merge(
            radiation_tensor.df.rename(
                columns={"K": "Kr", "Q": "Qr"},
            ),
            how="inner",
        )
        df_s["coefficient"] = df_s.t_s_1 * df_s.radiation_tensor
        self.matrix_builder.add_coefficient_from_df(df_s)

    def precompute_relaxation(self, level: Level, K: int, Q: int, J: float, Jʹ: float):
        """
        Reference: (7.38)
        """
        dfs_a = []
        dfs_e = []
        dfs_s = []

        # Relaxation from selected coherence
        for Jʹʹ, Jʹʹʹ, Kʹ, Qʹ in nested_loops(
            Jʹʹ=TRIANGULAR(level.L, level.S),
            Jʹʹʹ=TRIANGULAR(level.L, level.S),
            Kʹ=INTERSECTION(TRIANGULAR(J, Jʹ), TRIANGULAR("Jʹʹ", "Jʹʹʹ")),
            Qʹ=PROJECTION("Kʹ"),
        ):
            r_a_dfs = self.precompute_r_a(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            r_e_dfs = self.precompute_r_e(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)
            r_s_dfs = self.precompute_r_s(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ, Kʹ=Kʹ, Qʹ=Qʹ, Jʹʹ=Jʹʹ, Jʹʹʹ=Jʹʹʹ)

            r_a_dfs = [
                self.add_equation_index1(df, level_id="level_id", K="Kʹ", Q="Qʹ", J="Jʹʹ", Jʹ="Jʹʹʹ") for df in r_a_dfs
            ]
            r_e_dfs = [
                self.add_equation_index1(df, level_id="level_id", K="Kʹ", Q="Qʹ", J="Jʹʹ", Jʹ="Jʹʹʹ") for df in r_e_dfs
            ]
            r_s_dfs = [
                self.add_equation_index1(df, level_id="level_id", K="Kʹ", Q="Qʹ", J="Jʹʹ", Jʹ="Jʹʹʹ") for df in r_s_dfs
            ]
            dfs_a.extend(r_a_dfs)
            dfs_e.extend(r_e_dfs)
            dfs_s.extend(r_s_dfs)

        return dfs_a, dfs_e, dfs_s

    @log_method
    def add_precomputed_relaxation(self, df_a, df_e, df_s, radiation_tensor):
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
        self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float
    ):
        L = level.L
        S = level.S

        dfs = []
        for level_upper in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level_upper, level_lower=level):
                continue

            transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)
            Lu = level_upper.L

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
                            "level_id": [level.level_id],
                            "K": [K],
                            "Q": [Q],
                            "J": [J],
                            "Jʹ": [Jʹ],
                            "level_upper_id": [level_upper.level_id],
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
        self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float
    ):
        dfs = []
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level, level_lower=level_lower):
                continue
            transition = self.transition_registry.get_transition(level_upper=level, level_lower=level_lower)
            r_e_0 = delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ) * transition.einstein_a_ul
            dfs.append(
                pd.DataFrame(
                    {
                        "level_id": [level.level_id],
                        "K": [K],
                        "Q": [Q],
                        "J": [J],
                        "Jʹ": [Jʹ],
                        "level_lower_id": [level_lower.level_id],
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
        self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float
    ):
        # (7.46c)
        L = level.L
        S = level.S
        dfs = []
        for level_lower in self.term_registry.levels.values():
            if not self.transition_registry.is_transition_registered(level_upper=level, level_lower=level_lower):
                continue
            transition = self.transition_registry.get_transition(level_upper=level, level_lower=level_lower)
            Ll = level_lower.L
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
                )
                if self.disable_r_s:
                    r_s_1 = r_s_1 * 0

                dfs.append(
                    pd.DataFrame(
                        {
                            "level_id": [level.level_id],
                            "K": [K],
                            "Q": [Q],
                            "J": [J],
                            "Jʹ": [Jʹ],
                            "level_lower_id": [level_lower.level_id],
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
    def gamma(level: Level, J: float, Jʹ: float):
        """
        (7.42)
        """

        S = level.S
        L = level.L
        result = delta(J, Jʹ) * sqrt(J * (J + 1) * (2 * J + 1))
        result += (
            m1p(1 + L + S + J)
            * sqrt((2 * J + 1) * (2 * Jʹ + 1) * S * (S + 1) * (2 * S + 1))
            * wigner_6j(J, Jʹ, 1, S, S, L)
        )
        return result

    def precompute_n(
        self, level: Level, K: int, Q: int, J: float, Jʹ: float, Kʹ: int, Qʹ: int, Jʹʹ: float, Jʹʹʹ: float
    ):
        """
        Reference: (7.41)
        N = n_0 + n_1 * nu_larmor
        """

        term = self.term_registry.get_term(level=level, J=J)
        term_prime = self.term_registry.get_term(level=level, J=Jʹ)
        nu = energy_cmm1_to_frequency_hz(term.energy_cmm1 - term_prime.energy_cmm1)

        n_0 = delta(K, Kʹ) * delta(Q, Qʹ) * delta(J, Jʹʹ) * delta(Jʹ, Jʹʹʹ) * nu

        n_1 = (
            delta(Q, Qʹ)
            * m1p(J + Jʹ - Q)
            * sqrt((2 * K + 1) * (2 * Kʹ + 1))
            * wigner_3j(K, Kʹ, 1, -Q, Q, 0)
            * (
                delta(Jʹ, Jʹʹʹ) * self.gamma(level=level, J=J, Jʹ=Jʹʹ) * wigner_6j(K, Kʹ, 1, Jʹʹ, J, Jʹ)
                + delta(J, Jʹʹ)
                * m1p(K - Kʹ)
                * self.gamma(level=level, J=Jʹʹʹ, Jʹ=Jʹ)
                * wigner_6j(K, Kʹ, 1, Jʹʹʹ, Jʹ, J)
            )
        )
        if self.disable_n:
            n_0 = n_0 * 0
            n_1 = n_1 * 0

        df = pd.DataFrame(
            {
                "level_id": [level.level_id],
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
        level: Level,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        level_lower: Level,
        Kl: int,
        Ql: int,
        Jl: float,
        Jʹl: float,
    ):
        """
        T_a = t_a_1 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr)
        Reference: (7.45a)
        """
        S = level.S
        L = level.L
        Ll = level_lower.L

        transition = self.transition_registry.get_transition(level_upper=level, level_lower=level_lower)
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
            )
            dfs.append(
                pd.DataFrame(
                    {
                        "level_id": [level.level_id],
                        "K": [K],
                        "Q": [Q],
                        "J": [J],
                        "Jʹ": [Jʹ],
                        "level_lower_id": [level_lower.level_id],
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
        level: Level,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        level_upper: Level,
        Ku: int,
        Qu: int,
        Ju: float,
        Jʹu: float,
    ):
        """
        T_e = coefficient
        Reference: (7.45b)
        """

        S = level.S
        L = level.L
        Lu = level_upper.L

        transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)
        assert S == level_upper.S

        coefficient = multiply(
            lambda: delta(S, level_upper.S) * delta(K, Ku) * delta(Q, Qu),
            lambda: (2 * Lu + 1) * transition.einstein_a_ul,
            lambda: sqrt(n_proj(J, Jʹ, Ju, Jʹu)),
            lambda: m1p(1 + K + Jʹ + Jʹu),
            lambda: wigner_6j(J, Jʹ, K, Jʹu, Ju, 1),
            lambda: wigner_6j(Lu, L, 1, J, Ju, S),
            lambda: wigner_6j(Lu, L, 1, Jʹ, Jʹu, S),
        )

        return [
            pd.DataFrame(
                {
                    "level_id": [level.level_id],
                    "K": [K],
                    "Q": [Q],
                    "J": [J],
                    "Jʹ": [Jʹ],
                    "level_upper_id": [level_upper.level_id],
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
        level: Level,
        K: int,
        Q: int,
        J: float,
        Jʹ: float,
        level_upper: Level,
        Ku: int,
        Qu: int,
        Ju: float,
        Jʹu: float,
    ):
        """
        T_s = t_s_1 * self.radiation_tensor(transition=transition, K=Kr, Q=Qr)
        Reference: (7.45c)
        """
        S = level.S
        L = level.L
        Lu = level_upper.L

        transition = self.transition_registry.get_transition(level_upper=level_upper, level_lower=level)
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
            )

            dfs.append(
                pd.DataFrame(
                    {
                        "level_id": [level.level_id],
                        "K": [K],
                        "Q": [Q],
                        "J": [J],
                        "Jʹ": [Jʹ],
                        "level_upper_id": [level_upper.level_id],
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
    def get_solution_direct(self) -> Rho:
        """
        # A x = 0
        # let x[0] = 1 (it is always rho_0_0 of some kind, so it is unlikely to be exactly zero)
        # A[1:] x = 0
        # A[1:, 1:] x[1:] = -A[1:, 0]
        """
        logging.info("Get Solution of Statistical Equilibrium Equations")
        sol = np.linalg.solve(
            self.matrix_builder.rho_matrix[:, 1:, 1:],
            -self.matrix_builder.rho_matrix[:, 1:, 0:1],
        )
        sol = np.insert(sol, 0, 1.0, 1)
        sol = sol[:, :, 0]

        # Sum sqrt(2J+1) rho00(J, J) = 1
        weights = np.zeros_like(sol)
        for index, weight in zip(self.matrix_builder.trace_indexes, self.matrix_builder.trace_weights):
            weights[:, index] = weight
        trace = (sol * weights).sum(axis=1, keepdims=True)

        solution_vector = sol / trace

        rho = Rho(levels=list(self.term_registry.levels.values()))
        for index, (level_id, k, q, j, j_prime) in self.matrix_builder.index_to_parameters.items():
            rho.set_from_level_id(level_id=level_id, K=k, Q=q, J=j, Jʹ=j_prime, value=solution_vector[:, index])

        return rho

    def add_equation_index0(self, df: pd.DataFrame, level_id: str, K: str, Q: str, J: str, Jʹ: str):
        return self._add_equation_index(df=df, level_id=level_id, K=K, Q=Q, J=J, Jʹ=Jʹ, index="index0")

    def add_equation_index1(self, df: pd.DataFrame, level_id: str, K: str, Q: str, J: str, Jʹ: str):
        return self._add_equation_index(df=df, level_id=level_id, K=K, Q=Q, J=J, Jʹ=Jʹ, index="index1")

    def _add_equation_index(self, df: pd.DataFrame, level_id: str, K: str, Q: str, J: str, Jʹ: str, index: str):
        df[index] = df.apply(
            lambda row: self.matrix_builder.coherence_id_to_index[
                construct_coherence_id_from_level_id(level_id=row[level_id], K=row[K], Q=row[Q], J=row[J], Jʹ=row[Jʹ])
            ],
            axis=1,
        )
        return df
