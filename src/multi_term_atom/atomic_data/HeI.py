"""
TODO
TODO  This file needs improved documentation.
TODO
"""
from pathlib import Path

import pandas as pd

from src.engine.functions.decorators import log_function
from src.common.functions import lambda_cm_to_frequency_hz
from src.multi_term_atom.object.multi_term_atom_context import MultiTermAtomContext
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


@log_function
def get_He_I_D3_data():
    """
    Atomic model for He I atom including D3 transition.
    For details see A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
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

    # Reference lambda
    reference_lambda_A = 5877.25
    reference_nu_sm1 = lambda_cm_to_frequency_hz(reference_lambda_A * 1e-8)
    return level_registry, transition_registry, reference_lambda_A, reference_nu_sm1


def fill_precomputed_He_I_D3_data(
    atom: MultiTermAtomSEE,
    root="",
):
    directory = root + "/src/multi_term_atom/atomic_data/HeI_precomputed/"
    atom.coherence_decay_df = pd.read_parquet(directory + "coherence_decay_df.parquet")
    atom.absorption_df = pd.read_parquet(directory + "absorption_df.parquet")
    atom.emission_df_e = pd.read_parquet(directory + "emission_df_e.parquet")
    atom.emission_df_s = pd.read_parquet(directory + "emission_df_s.parquet")
    atom.relaxation_df_a = pd.read_parquet(directory + "relaxation_df_a.parquet")
    atom.relaxation_df_e = pd.read_parquet(directory + "relaxation_df_e.parquet")
    atom.relaxation_df_s = pd.read_parquet(directory + "relaxation_df_s.parquet")



def create_he_i_d3_context(lambda_range_A: float = 1.0, lambda_resolution_A: float = 1e-4) -> MultiTermAtomContext:
    """
    Create a MultiTermAtomContext for the He I D3 line (5877.25 Å).

    Args:
        lambda_range_A: Wavelength range around line center [Angstroms]
        lambda_resolution_A: Wavelength resolution [Angstroms]

    Returns:
        Configured MultiTermAtomContext for He I D3
    """
    # Get atomic data
    level_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()
    lambda_A = np.arange(reference_lambda_A - lambda_range_A, reference_lambda_A + lambda_range_A, lambda_resolution_A)

    # Set up statistical equilibrium equations
    see = MultiTermAtomSEE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        precompute=False,
    )

    # Load precomputed coefficients
    root_path = Path(__file__).resolve().parent.parent.parent.parent.as_posix()
    fill_precomputed_He_I_D3_data(see, root=root_path)

    context = MultiTermAtomContext(
        level_registry=level_registry,
        transition_registry=transition_registry,
        statistical_equilibrium_equations=see,
        lambda_A=lambda_A,
        reference_lambda_A=reference_lambda_A,
    )
    return context

