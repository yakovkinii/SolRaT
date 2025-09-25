import numpy as np
from pathlib import Path

from src.multi_term_atom.atomic_data.HeI import get_He_I_D3_data, fill_precomputed_He_I_D3_data
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE
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

    def __init__(self,
                 level_registry: LevelRegistry,
                 transition_registry: TransitionRegistry,
                 statistical_equilibrium_equations: MultiTermAtomSEE,
                 lambda_A: np.ndarray,
                 reference_lambda_A: float):
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
    lambda_A = np.arange(reference_lambda_A - lambda_range_A,
                        reference_lambda_A + lambda_range_A,
                        lambda_resolution_A)

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
