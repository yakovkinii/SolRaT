import logging
from typing import Optional, Union

from solrat.atom_model.base_atom_model.object.config import BaseConfig
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry


class MultiLevelAtomConfig(BaseConfig):
    r"""
    Configuration class for the Multi-Level Atom model.

    :param level_registry:  Registry of atomic levels.
    :param transition_registry:  Registry of radiative transitions.
    :param atomic_mass_amu:  Atomic mass [amu].
    :param reference_lambda_A_air:  Air reference wavelength (used for plotting only).
    :param disable_r_s:  DEPRECATED, scheduled for removal. Disables only the stimulated-emission
        relaxation term :math:`R_S`; it does NOT touch the transfer term :math:`T_S` or the
        stimulated-emission opacity :math:`\eta_S` in the RTE, so it removes stimulated emission only
        partially and inconsistently. To suppress stimulated emission physically, drive the model into
        the Wien limit instead (large :math:`h\nu_0 / k T`, i.e. high transition energy or low
        temperature), where the photon occupation number -- and hence all stimulated processes -- goes
        to zero self-consistently across the RTE and the SEE.
    :param custom_delta_nu_cutoff:  Distance in frequency for cutting off irrelevant transitions.
    :param N:  Atom numeric concentration for d/dz transfer modeling.
        Can be left equal to 1 for d/dtau modeling.
    :param collisions:  Optional :class:`ParametrizedCollisions` with parametrized collisional
        rates for the SEE. ``None`` means collisionless (pure scattering).
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
        collisions: Optional[ParametrizedCollisions] = None,
    ):
        self.level_registry = level_registry
        self.transition_registry = transition_registry
        self.atomic_mass_amu = atomic_mass_amu
        self.reference_lambda_A_air = reference_lambda_A_air
        self.disable_r_s = disable_r_s
        # DEPRECATED: disable_r_s is scheduled for removal. It gates only R_S (not T_S or the RTE
        # stimulated-emission opacity), so it is not a consistent way to switch stimulated emission off.
        if disable_r_s:
            logging.warning(
                "disable_r_s is deprecated and scheduled for removal: it disables only the R_S "
                "relaxation term, leaving the T_S transfer term and the RTE stimulated-emission opacity "
                "active, so stimulated emission is only partially removed. Use the Wien limit (large "
                "h*nu0 / kT) to suppress stimulated emission consistently instead."
            )
        self.custom_delta_nu_cutoff = custom_delta_nu_cutoff
        self.N = N
        self.collisions = collisions
