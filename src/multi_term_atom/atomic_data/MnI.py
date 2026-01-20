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
def get_Mn_I_5432_data():
    # Levels
    level_registry = LevelRegistry()

    # a 6S (L=0, S=2.5) J=2.5
    level_registry.register_level(beta="a6S", L=0, S=2.5, J=2.5, energy_cmm1=0)

    # z 8P° (L=1 S=3.5), J = 2.5, 3.5, 4.5
    # Within single-J approximation, intercombinational transition can
    # be modeled by pretending that the upper term has only allowed component
    # |upper> = cos theta z6P + sin theta z8P.
    # Here, sin theta is dominant, but cos theta is the only one allowed.
    # Also need to multiply RTE by cos^2 theta (assuming SEE is LTE)
    level_registry.register_level(beta="z6P+z8P", L=1, S=2.5, J=1.5, energy_cmm1=18000)  # Artificial, not coupled
    level_registry.register_level(beta="z6P+z8P", L=1, S=2.5, J=2.5, energy_cmm1=18402.46)  # Energy from 8P
    level_registry.register_level(beta="z6P+z8P", L=1, S=2.5, J=3.5, energy_cmm1=18531.64)  # Energy from 8P

    level_registry.validate()

    # Transitions
    transition_registry = TransitionRegistry()

    transition_registry.register_transition(
        term_lower=level_registry.get_term(beta="a6S", L=0, S=2.5),
        term_upper=level_registry.get_term(beta="z6P+z8P", L=1, S=2.5),
        lower_J_constraint=[2.5],  # used if j_constrained=True
        upper_J_constraint=[2.5],  # used if j_constrained=True
        einstein_a_ul_sm1=6.04e+03
    )

    reference_lambda_A = 5432.5
    reference_nu_sm1 = lambda_cm_to_frequency_hz(reference_lambda_A * 1e-8)
    return level_registry, transition_registry, reference_lambda_A, reference_nu_sm1
