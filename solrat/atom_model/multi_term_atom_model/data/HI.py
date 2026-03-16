from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import (
    MultiTermAtomConfig,
)
from solrat.atom_model.multi_term_atom_model.object.transition_registry import (
    TransitionRegistry,
)


def get_H_I_alpha_config() -> MultiTermAtomConfig:  # pragma: no cover
    r"""
    H alpha line (:math:`3\to 2`)
    Reference: NIST

    :return: :any:`MultiTermAtomConfig` instance
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
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3d2", L=2, S=0.5),
        term_lower=level_registry.get_term(beta="2p2", L=1, S=0.5),
        einstein_a_ul_sm1=5.3877e07 + 6.4651e07 + 1.0775e07,
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3p2", L=1, S=0.5),
        term_lower=level_registry.get_term(beta="2s2", L=0, S=0.5),
        einstein_a_ul_sm1=2.2448e07 + 2.2449e07,
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3s2", L=0, S=0.5),
        term_lower=level_registry.get_term(beta="2p2", L=1, S=0.5),
        einstein_a_ul_sm1=2.1046e06 + 4.2097e06,
    )

    # Reference lambda
    reference_lambda_A = 6563.3
    return MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        atomic_mass_amu=1,
        reference_lambda_A=reference_lambda_A,
    )
