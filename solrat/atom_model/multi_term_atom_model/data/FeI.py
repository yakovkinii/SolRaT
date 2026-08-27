from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.multi_term_atom_model.utility.paschen_back import get_artificial_S_scale_from_term_g
from solrat.engine.functions.decorators import log_function_experimental


@log_function_experimental
def get_Fe_I_5434_config() -> MultiTermAtomConfig:  # pragma: no cover
    r"""
    Atomic model for Fe I 5434.523 A line, constrained to :math:`J=0 \to J=1` transition.

    Additionally, the Lande factor is artificially deviated from pure LS coupling to model
    the experimental value of :math:`g=-0.014`.

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

    Due to rather crude approximations applied, this atomic model is recommended
    to be used with LTE SEE only.
    """
    level_registry = LevelRegistry()

    # --- lower term: a 5F (L=3, S=2), J = 1..5
    level_registry.register_level(beta="a5F", L=3, S=2, J=5, energy_cmm1=6928.268)
    level_registry.register_level(beta="a5F", L=3, S=2, J=4, energy_cmm1=7376.764)
    level_registry.register_level(beta="a5F", L=3, S=2, J=3, energy_cmm1=7728.060)
    level_registry.register_level(beta="a5F", L=3, S=2, J=2, energy_cmm1=7985.785)
    level_registry.register_level(beta="a5F", L=3, S=2, J=1, energy_cmm1=8154.714)

    # --- upper term: z 5D (L=2, S=2), J = 0..4
    level_registry.register_level(beta="z5Do", L=2, S=2, J=4, energy_cmm1=25899.989)
    level_registry.register_level(beta="z5Do", L=2, S=2, J=3, energy_cmm1=26140.179)
    level_registry.register_level(beta="z5Do", L=2, S=2, J=2, energy_cmm1=26339.696)
    level_registry.register_level(beta="z5Do", L=2, S=2, J=1, energy_cmm1=26479.381)
    level_registry.register_level(beta="z5Do", L=2, S=2, J=0, energy_cmm1=26550.479)

    level_registry.validate()
    level_registry.get_term(beta="a5F", L=3, S=2).set_artificial_spin_scale(
        get_artificial_S_scale_from_term_g(g=-0.014, L=3, S=2, J=1)
    )

    transition_registry = TransitionRegistry()

    transition_registry.register_transition(
        term_lower=level_registry.get_term(beta="a5F", L=3, S=2),
        term_upper=level_registry.get_term(beta="z5Do", L=2, S=2),
        lower_J_constraint=[1],  # Only compute J=0->J=1 in RTE (if j_constrained=True)
        upper_J_constraint=[0],
        einstein_a_ul_sm1=(
            1.70e6
            + 1.27e06
            + 1.15e06
            + 2.58e05
            + 1.05e06
            + 4.27e05
            + 2.20e04
            + 1.09e06
            + 5.48e05
            + 5.01e04
            + 6.05e05
            + 6.25e04
        )
        / 5,
    )

    return MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A_air=5434.523,
        atomic_mass_amu=55.8,
        j_constrained=True,
    )
