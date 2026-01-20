"""
TODO
TODO  This file needs improved documentation.
TODO
"""

import pandas as pd

from src.engine.functions.decorators import log_function
from src.common.functions import lambda_cm_to_frequency_hz
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


@log_function
def get_Ni_I_5435_data():
    # Levels
    level_registry = LevelRegistry()

    # --- lower term: 3P (L=1, S=1), J = 0..2
    # energies in cm^-1
    level_registry.register_level(beta="3P", L=1, S=1, J=0, energy_cmm1=16017.306)
    level_registry.register_level(beta="3P", L=1, S=1, J=1, energy_cmm1=15734.001)
    level_registry.register_level(beta="3P", L=1, S=1, J=2, energy_cmm1=15609.844)

    # --- upper term: 3D° (L=2, S=1), J = 1..3
    level_registry.register_level(beta="3D", L=2, S=1, J=1, energy_cmm1=34408.555   )
    level_registry.register_level(beta="3D", L=2, S=1, J=2, energy_cmm1=33610.890)
    level_registry.register_level(beta="3D", L=2, S=1, J=3, energy_cmm1= 33500.822 )

    level_registry.validate()

    # Transitions
    transition_registry = TransitionRegistry()

    # 0->1



    # Register the term-to-term transition with the provided Aki.
    # NOTE: Your current TransitionRegistry is term-based (not J-pair based).
    # The Aki=1.70e6 s^-1 is for the Ju=0 -> Jl=1 component.
    transition_registry.register_transition(
        term_lower=level_registry.get_term(beta="3P", L=1, S=1),
        term_upper=level_registry.get_term(beta="3D", L=2, S=1),
        lower_J_constraint=[0],
        upper_J_constraint=[1],
        einstein_a_ul_sm1=1.9e+05 +1.2e5 + 1.1e5 + 2.5e4 + 2.2e5
    )

    reference_lambda_A = 5435.9
    reference_nu_sm1 = lambda_cm_to_frequency_hz(reference_lambda_A * 1e-8)
    return level_registry, transition_registry, reference_lambda_A, reference_nu_sm1
