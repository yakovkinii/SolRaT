from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import (
    TransitionRegistry,
)


def get_mock_atom_data(fine_structure=True):
    """
    Ly-alpha-like structure for testing
    """
    # Levels
    level_registry = LevelRegistry()
    level_registry.register_level(
        beta="1s",
        L=0,
        S=0.5,
        J=0.5,
        energy_cmm1=200_000,
    )
    level_registry.register_level(
        beta="2p",
        L=1,
        S=0.5,
        J=0.5,
        energy_cmm1=220_000,
    )
    level_registry.register_level(
        beta="2p",
        L=1,
        S=0.5,
        J=1.5,
        energy_cmm1=220_001 if fine_structure else 220_000,
    )
    level_registry.validate()

    # Transitions

    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="2p", L=1, S=0.5),
        term_lower=level_registry.get_term(beta="1s", L=0, S=0.5),
        einstein_a_ul_sm1=0.7e8,
    )

    # Reference lambda
    reference_lambda_A = None
    reference_nu_sm1 = 5.996e14
    atomic_mass_amu = 1
    return level_registry, transition_registry, reference_lambda_A, reference_nu_sm1, atomic_mass_amu
