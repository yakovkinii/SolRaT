import pandas as pd

from src.engine.functions.decorators import log_function
from src.common.functions import lambda_cm_to_frequency_hz
from src.multi_term_atom.physics.paschen_back import get_artificial_S_scale_from_term_g
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


@log_function
def get_Fe_I_5434_data(scale_S=False):
    """
    Atomic model for Fe I 5434.523 Å component:
        lower: 3d7(4F)4s            a 5F   (L=3, S=2)  J=1..5
        upper: 3d6(5D)4s4p(3P°)     z 5D°  (L=2, S=2)  J=0..4
    with the specific observed component:
        Ju=0  ->  Jl=1

    Level energies are in cm^-1 (from the user-provided table).
    Aki corresponds to the Ju=0 -> Jl=1 component.

    # Scale_S means enforce experimental Lande factor for lower term.
    """

    # Levels
    level_registry = LevelRegistry()

    # --- lower term: a 5F (L=3, S=2), J = 1..5
    # energies in cm^-1
    level_registry.register_level(beta="a5F", L=3, S=2, J=5, energy_cmm1=6928.268)
    level_registry.register_level(beta="a5F", L=3, S=2, J=4, energy_cmm1=7376.764)
    level_registry.register_level(beta="a5F", L=3, S=2, J=3, energy_cmm1=7728.060)
    level_registry.register_level(beta="a5F", L=3, S=2, J=2, energy_cmm1=7985.785)
    level_registry.register_level(beta="a5F", L=3, S=2, J=1, energy_cmm1=8154.714)

    # --- upper term: z 5D° (L=2, S=2), J = 0..4
    level_registry.register_level(beta="z5Do", L=2, S=2, J=4, energy_cmm1=25899.989)
    level_registry.register_level(beta="z5Do", L=2, S=2, J=3, energy_cmm1=26140.179)
    level_registry.register_level(beta="z5Do", L=2, S=2, J=2, energy_cmm1=26339.696)
    level_registry.register_level(beta="z5Do", L=2, S=2, J=1, energy_cmm1=26479.381)
    level_registry.register_level(beta="z5Do", L=2, S=2, J=0, energy_cmm1=26550.479)

    level_registry.validate()
    level_registry.get_term(beta="a5F", L=3, S=2).set_artificial_S_scale(
        get_artificial_S_scale_from_term_g(g=-0.014, L=3, S=2, J=1)
    )

    # Transitions
    transition_registry = TransitionRegistry()

    # Register the term-to-term transition with the provided Aki.
    # NOTE: Your current TransitionRegistry is term-based (not J-pair based).
    # The Aki=1.70e6 s^-1 is for the Ju=0 -> Jl=1 component.
    transition_registry.register_transition_from_a_ul(
        term_lower=level_registry.get_term(beta="a5F", L=3, S=2),
        term_upper=level_registry.get_term(beta="z5Do", L=2, S=2),
        lower_J_for_RTE=[1],
        upper_J_for_RTE=[0],
        einstein_a_ul_sm1=1.70e6
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
        + 6.25e04,
    )

    reference_lambda_A = 5434.523
    reference_nu_sm1 = lambda_cm_to_frequency_hz(reference_lambda_A * 1e-8)
    return level_registry, transition_registry, reference_lambda_A, reference_nu_sm1
