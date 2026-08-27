from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.engine.functions.decorators import log_function


@log_function
def get_Ni_I_5435_config() -> MultiTermAtomConfig:  # pragma: no cover
    r"""
    Atomic model for Ni I 5435.9 A line, constrained to :math:`J=1 \to J=0` transition.

    ``einstein_a_ul_sm1`` is the LL04 term coefficient
    :math:`A(\beta_u L_u S \to \beta_l L_l S)`. NIST usually tabulates fine-structure components
    :math:`A(\beta_u L_u S J_u \to \beta_l L_l S J_l)`. To convert those component values to the
    SolRaT multi-term coefficient, fix one upper :math:`J_u` and sum over all lower :math:`J_l`.
    The result should be independent of the chosen :math:`J_u` in exact LS coupling. In practice,
    average those per-:math:`J_u` sums over all upper :math:`J_u` values included in the NIST
    component list:

    .. math::
        A_\mathrm{SolRaT}
        =
        \frac{1}{N_{J_u}}
        \sum_{J_u}
        \sum_{J_l}
        A_\mathrm{NIST}(J_u \to J_l).

    Here :math:`N_{J_u}` is just the number of distinct upper fine-structure :math:`J_u` values
    included for the upper term; it is not a separate NIST quantity. Do not use the raw sum over all
    fine-structure component lines unless the upper term has only one :math:`J_u`.

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
