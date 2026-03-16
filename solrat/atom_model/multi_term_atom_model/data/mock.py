from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import (
    MultiTermAtomConfig,
)
from solrat.atom_model.multi_term_atom_model.object.transition_registry import (
    TransitionRegistry,
)
from solrat.atom_model.shared.utility.functions import frequency_hz_to_lambda_A


def get_mock_atom_config(fine_structure: bool = True) -> MultiTermAtomConfig:  # pragma: no cover
    r"""
    Ly-alpha-like structure for testing

    :param fine_structure:  whether the mock model should include fine structure splitting
    :return: :any:`MultiTermAtomConfig` instance
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
        einstein_a_ul_sm1=1e8,
    )

    return MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A=frequency_hz_to_lambda_A(5.996e14),
        atomic_mass_amu=1.0,
    )
