from src.common.functions import lambda_cm_to_frequency_hz
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


def get_H_I_alpha_data():
    """
    H alpha line (3->2)
    Reference: NIST
    """

    # Levels
    level_registry = LevelRegistry()
    level_registry.register_level(
        beta="2s2",
        L=0,
        S=0.5,
        J=0.5,
        energy_cmm1=82258.9543992821,
    )
    level_registry.register_level(
        beta="2p2",
        L=1,
        S=0.5,
        J=0.5,
        energy_cmm1=82258.9191133,
    )
    level_registry.register_level(
        beta="2p2",
        L=1,
        S=0.5,
        J=1.5,
        energy_cmm1=82259.2850014,
    )

    level_registry.register_level(
        beta="3s2",
        L=0,
        S=0.5,
        J=0.5,
        energy_cmm1=97492.221701,
    )
    level_registry.register_level(
        beta="3p2",
        L=1,
        S=0.5,
        J=0.5,
        energy_cmm1=97492.211200,
    )
    level_registry.register_level(
        beta="3p2",
        L=1,
        S=0.5,
        J=1.5,
        energy_cmm1=97492.319611,
    )
    level_registry.register_level(
        beta="3d2",
        L=2,
        S=0.5,
        J=1.5,
        energy_cmm1=97492.319433,
    )
    level_registry.register_level(
        beta="3d2",
        L=2,
        S=0.5,
        J=2.5,
        energy_cmm1=97492.355566,
    )

    level_registry.validate()

    # Transitions
    transition_registry = TransitionRegistry()
    transition_registry.register_transition_from_a_ul(
        term_upper=level_registry.get_term(beta="3d2", L=2, S=0.5),
        term_lower=level_registry.get_term(beta="2p2", L=1, S=0.5),
        einstein_a_ul_sm1=5.3877e07 + 6.4651e07 + 1.0775e07,
    )
    transition_registry.register_transition_from_a_ul(
        term_upper=level_registry.get_term(beta="3p2", L=1, S=0.5),
        term_lower=level_registry.get_term(beta="2s2", L=0, S=0.5),
        einstein_a_ul_sm1=2.2448e07 + 2.2449e07,
    )
    transition_registry.register_transition_from_a_ul(
        term_upper=level_registry.get_term(beta="3s2", L=0, S=0.5),
        term_lower=level_registry.get_term(beta="2p2", L=1, S=0.5),
        einstein_a_ul_sm1=2.1046e06 + 4.2097e06,
    )

    # Reference lambda
    reference_lambda_A = 6563.3
    reference_nu_sm1 = lambda_cm_to_frequency_hz(reference_lambda_A * 1e-8)
    return level_registry, transition_registry, reference_lambda_A, reference_nu_sm1
