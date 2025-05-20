import pandas as pd

from src.core.engine.functions.decorators import log_function
from src.core.physics.functions import lambda_cm_to_frequency_hz
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtomSEE
from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


@log_function
def get_He_I_D3_data():
    """
    Atomic model for He I atom including D3 transition.
    For details see A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
    """

    # Terms
    term_registry = TermRegistry()
    term_registry.register_term(
        beta="2s3",
        L=0,
        S=1,
        J=1,
        energy_cmm1=159855.9726,
    )
    term_registry.register_term(
        beta="3s3",
        L=0,
        S=1,
        J=1,
        energy_cmm1=183236.7905,
    )
    term_registry.register_term(
        beta="2p3",
        L=1,
        S=1,
        J=0,
        energy_cmm1=169087.8291,
    )
    term_registry.register_term(
        beta="2p3",
        L=1,
        S=1,
        J=1,
        energy_cmm1=169086.8412,
    )
    term_registry.register_term(
        beta="2p3",
        L=1,
        S=1,
        J=2,
        energy_cmm1=169086.7647,
    )

    term_registry.register_term(
        beta="3p3",
        L=1,
        S=1,
        J=0,
        energy_cmm1=185564.8528,
    )
    term_registry.register_term(
        beta="3p3",
        L=1,
        S=1,
        J=1,
        energy_cmm1=185564.5817,
    )
    term_registry.register_term(
        beta="3p3",
        L=1,
        S=1,
        J=2,
        energy_cmm1=185564.5602,
    )
    term_registry.register_term(
        beta="3d3",
        L=2,
        S=1,
        J=1,
        energy_cmm1=186101.5908,
    )
    term_registry.register_term(
        beta="3d3",
        L=2,
        S=1,
        J=2,
        energy_cmm1=186101.5466,
    )
    term_registry.register_term(
        beta="3d3",
        L=2,
        S=1,
        J=3,
        energy_cmm1=186101.5440,
    )
    term_registry.validate()

    # Transitions
    transition_registry = TransitionRegistry()
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="2p3", L=1, S=1),
        level_lower=term_registry.get_level(beta="2s3", L=0, S=1),
        einstein_a_ul_sm1=3 * 1.022e7,
    )
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="3p3", L=1, S=1),
        level_lower=term_registry.get_level(beta="2s3", L=0, S=1),
        einstein_a_ul_sm1=3 * 9.478e6,
    )
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="3s3", L=0, S=1),
        level_lower=term_registry.get_level(beta="2p3", L=1, S=1),
        einstein_a_ul_sm1=3.080e6 + 9.259e6 + 1.540e7,
    )
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="3d3", L=2, S=1),
        level_lower=term_registry.get_level(beta="2p3", L=1, S=1),
        einstein_a_ul_sm1=3.920e7 + 5.290e7 + 2.940e7 + 7.060e7 + 1.760e7 + 1.960e6,
    )

    # Reference lambda
    reference_lambda_A = 5877.25
    reference_nu_sm1 = lambda_cm_to_frequency_hz(reference_lambda_A * 1e-8)
    return term_registry, transition_registry, reference_lambda_A, reference_nu_sm1


def fill_precomputed_He_I_D3_data(
    atom: TwoTermAtomSEE,
    root="",
):
    directory = root + "src/two_term_atom/atomic_data/HeI_precomputed/"
    atom.coherence_decay_df = pd.read_parquet(directory + "coherence_decay_df.parquet")
    atom.absorption_df = pd.read_parquet(directory + "absorption_df.parquet")
    atom.emission_df_e = pd.read_parquet(directory + "emission_df_e.parquet")
    atom.emission_df_s = pd.read_parquet(directory + "emission_df_s.parquet")
    atom.relaxation_df_a = pd.read_parquet(directory + "relaxation_df_a.parquet")
    atom.relaxation_df_e = pd.read_parquet(directory + "relaxation_df_e.parquet")
    atom.relaxation_df_s = pd.read_parquet(directory + "relaxation_df_s.parquet")
