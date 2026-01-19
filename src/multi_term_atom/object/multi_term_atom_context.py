"""
TODO
TODO  This file needs improved documentation.
TODO
"""

from typing import Union, Tuple

import numpy as np
from pathlib import Path

from src.multi_term_atom.atomic_data.FeI import get_Fe_I_5434_data
from src.multi_term_atom.atomic_data.HeI import get_He_I_D3_data, fill_precomputed_He_I_D3_data
from src.multi_term_atom.atomic_data.MnI import get_Mn_I_5432_data
from src.multi_term_atom.atomic_data.NiI import get_Ni_I_5435_data
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE, MultiTermAtomSEELTE
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class MultiTermAtomContext:
    """
    Container class that holds all the atomic structure information and
    statistical equilibrium equations for a multi-term atom model.

    Attributes:
        level_registry: Registry of atomic energy levels
        transition_registry: Registry of radiative transitions
        statistical_equilibrium_equations: Precomputed SEE solver
        lambda_A: Wavelength grid [Angstroms]
        reference_lambda_A: Reference wavelength [Angstroms]
    """

    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        statistical_equilibrium_equations: Union[MultiTermAtomSEE, MultiTermAtomSEELTE],
        lambda_A: np.ndarray,
        reference_lambda_A: float,
    ):
        self.level_registry = level_registry
        self.transition_registry = transition_registry
        self.statistical_equilibrium_equations = statistical_equilibrium_equations
        self.lambda_A = lambda_A
        self.reference_lambda_A = reference_lambda_A


def create_he_i_d3_context(lambda_range_A: float = 1.0, lambda_resolution_A: float = 1e-4) -> MultiTermAtomContext:
    """
    Create a MultiTermAtomContext for the He I D3 line (5877.25 Å).

    Args:
        lambda_range_A: Wavelength range around line center [Angstroms]
        lambda_resolution_A: Wavelength resolution [Angstroms]

    Returns:
        Configured MultiTermAtomContext for He I D3
    """
    # Get atomic data
    level_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()
    lambda_A = np.arange(reference_lambda_A - lambda_range_A, reference_lambda_A + lambda_range_A, lambda_resolution_A)

    # Set up statistical equilibrium equations
    see = MultiTermAtomSEE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        precompute=False,
    )

    # Load precomputed coefficients
    root_path = Path(__file__).resolve().parent.parent.parent.parent.as_posix()
    fill_precomputed_He_I_D3_data(see, root=root_path)

    context = MultiTermAtomContext(
        level_registry=level_registry,
        transition_registry=transition_registry,
        statistical_equilibrium_equations=see,
        lambda_A=lambda_A,
        reference_lambda_A=reference_lambda_A,
    )
    return context



def create_5434_MnFeNi_context(lambda_range_A: float = 1.0, lambda_resolution_A: float = 1e-4) -> Tuple[MultiTermAtomContext, MultiTermAtomContext,MultiTermAtomContext]:
    # Get atomic data
    level_registry_Mn, transition_registry_Mn, reference_lambda_A_Mn, _ = get_Mn_I_5432_data()
    level_registry_Fe, transition_registry_Fe, reference_lambda_A_Fe, _ = get_Fe_I_5434_data()
    level_registry_Ni, transition_registry_Ni, reference_lambda_A_Ni, _ = get_Ni_I_5435_data()


    lambda_A = np.arange(min(
        reference_lambda_A_Fe
        ,reference_lambda_A_Mn,reference_lambda_A_Ni
    ) - lambda_range_A, max(
        reference_lambda_A_Fe
        ,reference_lambda_A_Mn,reference_lambda_A_Ni
    ) + lambda_range_A, lambda_resolution_A)

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
