"""
TODO
TODO  This file needs improved documentation.
TODO
"""
from typing import Tuple

import pandas as pd

from src.common.functions import lambda_cm_to_frequency_hz
from src.engine.functions.decorators import log_function
from src.multi_term_atom.object.multi_term_atom_context import MultiTermAtomContext
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import (
    TransitionRegistry,
)


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
    level_registry.register_level(beta="3D", L=2, S=1, J=1, energy_cmm1=34408.555)
    level_registry.register_level(beta="3D", L=2, S=1, J=2, energy_cmm1=33610.890)
    level_registry.register_level(beta="3D", L=2, S=1, J=3, energy_cmm1=33500.822)

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
        einstein_a_ul_sm1=1.9e05 + 1.2e5 + 1.1e5 + 2.5e4 + 2.2e5,
    )

    reference_lambda_A = 5435.9
    reference_nu_sm1 = lambda_cm_to_frequency_hz(reference_lambda_A * 1e-8)
    atomic_mass_amu = 58.7
    return level_registry, transition_registry, reference_lambda_A, reference_nu_sm1, atomic_mass_amu


def create_5434_MnFeNi_context(
    lambda_range_A: float = 1.0, lambda_resolution_A: float = 1e-4
) -> Tuple[MultiTermAtomContext, MultiTermAtomContext, MultiTermAtomContext]:
    # Get atomic data
    level_registry_Mn, transition_registry_Mn, reference_lambda_A_Mn, _ = get_Mn_I_5432_data()
    level_registry_Fe, transition_registry_Fe, reference_lambda_A_Fe, _ = get_Fe_I_5434_data()
    level_registry_Ni, transition_registry_Ni, reference_lambda_A_Ni, _ = get_Ni_I_5435_data()

    lambda_A = np.arange(
        min(reference_lambda_A_Fe, reference_lambda_A_Mn, reference_lambda_A_Ni) - lambda_range_A,
        max(reference_lambda_A_Fe, reference_lambda_A_Mn, reference_lambda_A_Ni) + lambda_range_A,
        lambda_resolution_A,
    )

    lambda_A = lambda_A + 1.5  # vac -> air

    # Set up statistical equilibrium equations
    see_Mn = MultiTermAtomSEELTE(
        level_registry=level_registry_Mn,
        atomic_mass_amu=54.9,
    )
    see_Fe = MultiTermAtomSEELTE(
        level_registry=level_registry_Fe,
        atomic_mass_amu=55.8,
    )
    see_Ni = MultiTermAtomSEELTE(
        level_registry=level_registry_Ni,
        atomic_mass_amu=58.7,
    )

    context_Mn = MultiTermAtomContext(
        level_registry=level_registry_Mn,
        transition_registry=transition_registry_Mn,
        statistical_equilibrium_equations=see_Mn,
        lambda_A=lambda_A,
        reference_lambda_A=reference_lambda_A_Fe,
    )
    context_Fe = MultiTermAtomContext(
        level_registry=level_registry_Fe,
        transition_registry=transition_registry_Fe,
        statistical_equilibrium_equations=see_Fe,
        lambda_A=lambda_A,
        reference_lambda_A=reference_lambda_A_Fe,
    )
    context_Ni = MultiTermAtomContext(
        level_registry=level_registry_Ni,
        transition_registry=transition_registry_Ni,
        statistical_equilibrium_equations=see_Ni,
        lambda_A=lambda_A,
        reference_lambda_A=reference_lambda_A_Fe,
    )
    return context_Mn, context_Fe, context_Ni
