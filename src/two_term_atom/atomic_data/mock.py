from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


def get_mock_atom_data(fine_structure=True):
    """
    Ly-alpha-like structure for testing
    """
    # Terms
    term_registry = TermRegistry()
    term_registry.register_term(
        beta="1s",
        L=0,
        S=0.5,
        J=0.5,
        energy_cmm1=200_000,
    )
    term_registry.register_term(
        beta="2p",
        L=1,
        S=0.5,
        J=0.5,
        energy_cmm1=220_000,
    )
    term_registry.register_term(
        beta="2p",
        L=1,
        S=0.5,
        J=1.5,
        energy_cmm1=220_001 if fine_structure else 220_000,
    )
    term_registry.validate()

    # Transitions

    transition_registry = TransitionRegistry()
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="2p", L=1, S=0.5),
        level_lower=term_registry.get_level(beta="1s", L=0, S=0.5),
        einstein_a_ul_sm1=0.7e8,
    )

    # Reference lambda
    reference_lambda_A = None
    reference_nu_sm1 = 5.996e14
    return term_registry, transition_registry, reference_lambda_A, reference_nu_sm1
