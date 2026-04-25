from typing import Union

from solrat.atom_model.base_atom_model.object.config import BaseConfig
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry


class MultiLevelAtomConfig(BaseConfig):
    r"""
    Configuration class for the Multi-Level Atom model.

    :param level_registry:  Registry of atomic levels.
    :param transition_registry:  Registry of radiative transitions.
    :param atomic_mass_amu:  Atomic mass [amu].
    :param reference_lambda_A_air:  Air reference wavelength (used for plotting only).
    :param disable_r_s:  Whether to disable stimulated emission relaxation :math:`R_S`.
    :param custom_delta_nu_cutoff:  Distance in frequency for cutting off irrelevant transitions.
    :param N:  Atom numeric concentration for d/dz transfer modeling.
        Can be left equal to 1 for d/dtau modeling.
    """

    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        atomic_mass_amu: float,
        reference_lambda_A_air: float,
        disable_r_s: bool = False,
        custom_delta_nu_cutoff: Union[float, None] = None,
        N: float = 1.0,
    ):
        self.level_registry = level_registry
        self.transition_registry = transition_registry
        self.atomic_mass_amu = atomic_mass_amu
        self.reference_lambda_A_air = reference_lambda_A_air
        self.disable_r_s = disable_r_s
        self.custom_delta_nu_cutoff = custom_delta_nu_cutoff
        self.N = N
