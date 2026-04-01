from typing import Union

from solrat.atom_model.base_atom_model.object.config import BaseConfig
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.precomputed_data import PrecomputedData
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry


class MultiTermAtomConfig(BaseConfig):
    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        atomic_mass_amu: float,
        reference_lambda_A: float,
        j_constrained=False,
        disable_r_s: bool = False,
        precomputed_data: Union[PrecomputedData, None] = None,
        custom_delta_nu_cutoff: Union[float, None] = None,
        N: float = 1.0,
    ):
        r"""
        Configuration class that specifies the atomic structure for the Multi-Term Atom model.

        :param level_registry: Registry of atomic energy levels
        :param transition_registry: Registry of radiative transitions
        :param atomic_mass_amu: Atomic mass (in atomic mass units)
        :param j_constrained: Enable :math:`J` constraint for selecting possible transitions in RTE
        (if constraint is specified in transition registry)
        """

        self.level_registry = level_registry
        self.transition_registry = transition_registry
        self.reference_lambda_A = reference_lambda_A
        self.j_constrained = j_constrained
        self.atomic_mass_amu = atomic_mass_amu
        self.disable_r_s = disable_r_s
        self.precomputed_data = precomputed_data
        self.custom_delta_nu_cutoff = custom_delta_nu_cutoff
        self.N = N
