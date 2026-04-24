from pathlib import Path

import pandas as pd

from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.precomputed_data import PrecomputedData
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.engine.functions.decorators import log_function


@log_function
def get_He_I_D3_config() -> MultiTermAtomConfig:  # pragma: no cover
    r"""
    Config constructor for He I atom, with multiple transitions including D3.
    For details see A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433

    :return: :any:`MultiTermAtomConfig` instance
    """

    # Levels
    level_registry = LevelRegistry()
    level_registry.register_level(
        beta="2s3",
        L=0,
        S=1,
        J=1,
        energy_cmm1=159855.9726,
    )
    level_registry.register_level(
        beta="3s3",
        L=0,
        S=1,
        J=1,
        energy_cmm1=183236.7905,
    )
    level_registry.register_level(
        beta="2p3",
        L=1,
        S=1,
        J=0,
        energy_cmm1=169087.8291,
    )
    level_registry.register_level(
        beta="2p3",
        L=1,
        S=1,
        J=1,
        energy_cmm1=169086.8412,
    )
    level_registry.register_level(
        beta="2p3",
        L=1,
        S=1,
        J=2,
        energy_cmm1=169086.7647,
    )

    level_registry.register_level(
        beta="3p3",
        L=1,
        S=1,
        J=0,
        energy_cmm1=185564.8528,
    )
    level_registry.register_level(
        beta="3p3",
        L=1,
        S=1,
        J=1,
        energy_cmm1=185564.5817,
    )
    level_registry.register_level(
        beta="3p3",
        L=1,
        S=1,
        J=2,
        energy_cmm1=185564.5602,
    )
    level_registry.register_level(
        beta="3d3",
        L=2,
        S=1,
        J=1,
        energy_cmm1=186101.5908,
    )
    level_registry.register_level(
        beta="3d3",
        L=2,
        S=1,
        J=2,
        energy_cmm1=186101.5466,
    )
    level_registry.register_level(
        beta="3d3",
        L=2,
        S=1,
        J=3,
        energy_cmm1=186101.5440,
    )
    level_registry.validate()

    # Transitions
    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="2p3", L=1, S=1),
        term_lower=level_registry.get_term(beta="2s3", L=0, S=1),
        einstein_a_ul_sm1=3 * 1.022e7,
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3p3", L=1, S=1),
        term_lower=level_registry.get_term(beta="2s3", L=0, S=1),
        einstein_a_ul_sm1=3 * 9.478e6,
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3s3", L=0, S=1),
        term_lower=level_registry.get_term(beta="2p3", L=1, S=1),
        einstein_a_ul_sm1=3.080e6 + 9.259e6 + 1.540e7,
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3d3", L=2, S=1),
        term_lower=level_registry.get_term(beta="2p3", L=1, S=1),
        einstein_a_ul_sm1=3.920e7 + 5.290e7 + 2.940e7 + 7.060e7 + 1.760e7 + 1.960e6,
    )
    root = Path(__file__).resolve().parent.as_posix()
    directory = root + "/HeI_precomputed/"
    precomputed_data = PrecomputedData(
        coherence_decay_df=pd.read_csv(directory + "coherence_decay_df.csv"),
        absorption_df=pd.read_csv(directory + "absorption_df.csv"),
        emission_df_e=pd.read_csv(directory + "emission_df_e.csv"),
        emission_df_s=pd.read_csv(directory + "emission_df_s.csv"),
        relaxation_df_a=pd.read_csv(directory + "relaxation_df_a.csv"),
        relaxation_df_e=pd.read_csv(directory + "relaxation_df_e.csv"),
        relaxation_df_s=pd.read_csv(directory + "relaxation_df_s.csv"),
    )

    return MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A_air=5875.621,
        atomic_mass_amu=4.0,
        precomputed_data=precomputed_data,
    )
