import logging
from typing import Union

from solrat.atom_model.base_atom_model.object.config import BaseConfig
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.precomputed_data import PrecomputedData
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry


class MultiTermAtomConfig(BaseConfig):
    r"""
    Configuration class that specifies the atomic structure for the Multi-Term Atom model.

    :param level_registry: Registry of atomic energy levels
    :param transition_registry: Registry of radiative transitions
    :param atomic_mass_amu: Atomic mass (in atomic mass units)
    :param reference_lambda_A_air: air reference wavelength (used for plotting only)
    :param j_constrained: Enable :math:`J` constraint for selecting possible transitions in RTE
        (if constraint is specified in transition registry)
    :param disable_r_s: DEPRECATED, scheduled for removal. Disables only the stimulated-emission
        relaxation term :math:`R_S`; it does NOT touch the transfer term :math:`T_S` or the
        stimulated-emission opacity :math:`\eta_S` in the RTE, so it removes stimulated emission only
        partially and inconsistently. To suppress stimulated emission physically, drive the model into
        the Wien limit instead (large :math:`h\nu_0 / k T`), where the photon occupation number -- and
        hence all stimulated processes -- goes to zero self-consistently across the RTE and the SEE.
    """

    def __init__(
        self,
        level_registry: LevelRegistry,
        transition_registry: TransitionRegistry,
        atomic_mass_amu: float,
        reference_lambda_A_air: float,
        j_constrained=False,
        disable_r_s: bool = False,
        precomputed_data: Union[PrecomputedData, None] = None,
        custom_delta_nu_cutoff: Union[float, None] = None,
        N: float = 1.0,
        collisions=None,
    ):
        self.level_registry = level_registry
        self.transition_registry = transition_registry
        self.reference_lambda_A_air = reference_lambda_A_air
        self.j_constrained = j_constrained
        self.atomic_mass_amu = atomic_mass_amu
        self.disable_r_s = disable_r_s
        # Optional parametrized collisional rates (ParametrizedCollisions); None means collisionless
        # (pure scattering). Supports terms with any number of J levels; see SEE.add_collisions.
        self.collisions = collisions
        # DEPRECATED: disable_r_s is scheduled for removal. It gates only R_S (not T_S or the RTE
        # stimulated-emission opacity), so it is not a consistent way to switch stimulated emission off.
        if disable_r_s:
            logging.warning(
                "disable_r_s is deprecated and scheduled for removal: it disables only the R_S "
                "relaxation term, leaving the T_S transfer term and the RTE stimulated-emission opacity "
                "active, so stimulated emission is only partially removed. Use the Wien limit (large "
                "h*nu0 / kT) to suppress stimulated emission consistently instead."
            )
        self.precomputed_data = precomputed_data
        self.custom_delta_nu_cutoff = custom_delta_nu_cutoff
        self.N = N
