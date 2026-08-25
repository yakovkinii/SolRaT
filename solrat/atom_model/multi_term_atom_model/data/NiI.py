from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.engine.functions.decorators import log_function


@log_function
def get_Ni_I_5435_config() -> MultiTermAtomConfig:  # pragma: no cover
    r"""
    Atomic model for Ni I 5435.9 A line, constrained to :math:`J=1 \to J=0` transition.

    :return: :any:`MultiTermAtomConfig` instance
    """

    level_registry = LevelRegistry()

    # lower term: 3P (L=1, S=1), J = 0..2
    level_registry.register_level(beta="3P", L=1, S=1, J=0, energy_cmm1=16017.306)
    level_registry.register_level(beta="3P", L=1, S=1, J=1, energy_cmm1=15734.001)
    level_registry.register_level(beta="3P", L=1, S=1, J=2, energy_cmm1=15609.844)

    # upper term: 3D (L=2, S=1), J = 1..3
    level_registry.register_level(beta="3D", L=2, S=1, J=1, energy_cmm1=34408.555)
    level_registry.register_level(beta="3D", L=2, S=1, J=2, energy_cmm1=33610.890)
    level_registry.register_level(beta="3D", L=2, S=1, J=3, energy_cmm1=33500.822)

    level_registry.validate()

    # Transitions
    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        term_lower=level_registry.get_term(beta="3P", L=1, S=1),
        term_upper=level_registry.get_term(beta="3D", L=2, S=1),
        lower_J_constraint=[0],  # Only compute J=1->J=0 in RTE (if j_constrained=True)
        upper_J_constraint=[1],
        einstein_a_ul_sm1=(1.9e05 + 1.2e5 + 1.1e5 + 2.5e4 + 2.2e5) / 3,
    )

    return MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A_air=5435.87,
        atomic_mass_amu=58.7,
        j_constrained=True,
    )
