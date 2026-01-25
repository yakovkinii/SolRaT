from typing import Union

import numpy as np

from src.common.functions import lambda_cm_to_frequency_hz
from src.multi_term_atom.statistical_equilibrium_equations import (
    MultiTermAtomSEE,
    MultiTermAtomSEELTE,
)
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import (
    TransitionRegistry,
)


class MultiTermAtomContext:
    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        statistical_equilibrium_equations: Union[MultiTermAtomSEE, MultiTermAtomSEELTE],
        lambda_A: np.ndarray,
        reference_lambda_A: float,
        atomic_mass_amu: float,
        j_constrained=False,
    ):
        """
        Container class that holds all the atomic structure information and
        statistical equilibrium equations for a multi-term atom model.

        :param level_registry: Registry of atomic energy levels
        :param transition_registry: Registry of radiative transitions
        :param statistical_equilibrium_equations: SEE/SEELTE object
        :param lambda_A: Wavelength grid
        :param reference_lambda_A: Reference wavelength (for plotting etc.)
        :param atomic_mass_amu: Atomic mass (in atomic mass units)
        :param j_constrained: Enable J constraint for selecting possible transitions in RTE
        (if constraint is specified in transition registry)
        """

        self.level_registry = level_registry
        self.transition_registry = transition_registry
        self.statistical_equilibrium_equations = statistical_equilibrium_equations
        self.lambda_A = lambda_A
        self.reference_lambda_A = reference_lambda_A
        self.j_constrained = j_constrained
        self.atomic_mass_amu = atomic_mass_amu
        self.nu = lambda_cm_to_frequency_hz(self.lambda_A * 1e-8)
