from typing import Union

import numpy as np

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
    ):
        """
        Container class that holds all the atomic structure information and
        statistical equilibrium equations for a multi-term atom model.

        :param level_registry: Registry of atomic energy levels
        :param transition_registry: Registry of radiative transitions
        :param statistical_equilibrium_equations: SEE/SEELTE object
        :param lambda_A: Wavelength grid
        :param reference_lambda_A: Reference wavelength (for plotting etc.)
        """

        self.level_registry = level_registry
        self.transition_registry = transition_registry
        self.statistical_equilibrium_equations = statistical_equilibrium_equations
        self.lambda_A = lambda_A
        self.reference_lambda_A = reference_lambda_A
