from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.engine.functions.decorators import log_function


@log_function
def get_H_I_alpha_config() -> MultiTermAtomConfig:  # pragma: no cover
    r"""
    H alpha line (:math:`3\to 2`)
    Reference: NIST

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
        einstein_a_ul_sm1=(5.3877e07 + 6.4651e07 + 1.0775e07) / 2,
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3p2", L=1, S=0.5),
        term_lower=level_registry.get_term(beta="2s2", L=0, S=0.5),
        einstein_a_ul_sm1=(2.2448e07 + 2.2449e07) / 2,
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3s2", L=0, S=0.5),
        term_lower=level_registry.get_term(beta="2p2", L=1, S=0.5),
        einstein_a_ul_sm1=2.1046e06 + 4.2097e06,
    )

    return MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        atomic_mass_amu=1,
        reference_lambda_A_air=6562.79,
    )
